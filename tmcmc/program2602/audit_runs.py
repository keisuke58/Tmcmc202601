#!/usr/bin/env python3
"""
Audit TMCMC runs under tmcmc/_runs.

Classifies each run directory into:
  - SUCCESS: last recorded subprocess rc==0 (or REPORT.md exists and no rc found)
  - FAIL: last recorded subprocess rc!=0 OR traceback detected in last segment
  - INCOMPLETE: subprocess started but no END rc recorded
  - UNKNOWN: no subprocess.log and no REPORT.md

Outputs:
  - Markdown summary + table (default: print to stdout)
  - Optional CSV for machine use

Optionally isolates failures by moving them to _runs/_failed (dry-run by default).
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


RUN_ID_RE = re.compile(r"^sig(?P<sig>\d{4})_cov(?P<cov>\d{4})_np(?P<np>\d{4})_ns(?P<ns>\d{2})$")
END_RE = re.compile(r"^END\s+(?P<ts>.+?)\s+rc=(?P<rc>-?\d+)\s*$")
START_RE = re.compile(r"^START\s+(?P<ts>.+?)\s*$")
CMD_RE = re.compile(r"^CMD:\s+(?P<cmd>.+?)\s*$")
REPORT_STATUS_RE = re.compile(r"^- \*\*status\*\*: \*\*(?P<status>PASS|WARN|FAIL)\*\*", re.MULTILINE)
TRACEBACK_MARK = "Traceback (most recent call last):"
EXC_RE = re.compile(r"^(?P<etype>\w+Error|Exception)\s*:\s*(?P<msg>.*)\s*$")


@dataclass(frozen=True)
class RunAudit:
    rel_path: str
    run_id: str
    has_report: bool
    report_status: str
    has_subprocess_log: bool
    last_rc: Optional[int]
    last_end_ts: str
    last_cmd: str
    status: str
    error: str
    params: str


def _safe_read_text(path: Path, max_bytes: int = 5_000_000) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()
    if len(data) > max_bytes:
        data = data[-max_bytes:]  # tail
    return data.decode("utf-8", errors="replace")


def _parse_report_status(report_path: Path) -> str:
    txt = _safe_read_text(report_path, max_bytes=1_000_000)
    m = REPORT_STATUS_RE.search(txt)
    return m.group("status") if m else "missing"


def _format_params_from_run_id(run_id: str) -> str:
    m = RUN_ID_RE.match(run_id)
    if not m:
        return "—"
    sig = int(m.group("sig")) / 1000.0
    cov = int(m.group("cov")) / 1000.0
    np_ = int(m.group("np"))
    ns = int(m.group("ns"))
    return f"sigma_obs={sig:g}, cov_rel={cov:g}, n_particles={np_}, n_stages={ns}"


def _parse_subprocess_log(log_path: Path) -> tuple[Optional[int], str, str, str, bool]:
    """
    Returns:
      last_rc, last_end_ts, last_cmd, last_error, traceback_in_last_segment
    """
    txt = _safe_read_text(log_path)
    if not txt:
        return None, "missing", "missing", "missing", False

    lines = txt.splitlines()
    last_start_i: Optional[int] = None
    last_cmd: str = "missing"
    last_end_ts: str = "missing"
    last_rc: Optional[int] = None
    last_error: str = "missing"
    traceback_in_last_segment = False

    # Track "segments" between START..END; we only care about the last segment.
    for i, line in enumerate(lines):
        sm = START_RE.match(line)
        if sm:
            last_start_i = i
            # reset per-segment signals
            traceback_in_last_segment = False
            last_cmd = "missing"
            last_error = "missing"
            continue

        cm = CMD_RE.match(line)
        if cm and last_start_i is not None:
            last_cmd = cm.group("cmd")
            continue

        if last_start_i is not None and line.strip() == TRACEBACK_MARK:
            traceback_in_last_segment = True
            continue

        if last_start_i is not None:
            em = EXC_RE.match(line.strip())
            if em:
                etype = em.group("etype")
                msg = em.group("msg").strip()
                last_error = f"{etype}: {msg}" if msg else etype

        endm = END_RE.match(line)
        if endm:
            last_end_ts = endm.group("ts")
            try:
                last_rc = int(endm.group("rc"))
            except Exception:
                last_rc = None
            # END closes current segment; keep last_start_i as-is, but next START will reset flags.

    return last_rc, last_end_ts, last_cmd, last_error, traceback_in_last_segment


def _classify(run_dir: Path, runs_root: Path) -> RunAudit:
    rel = run_dir.relative_to(runs_root).as_posix()
    run_id = run_dir.name
    report = run_dir / "REPORT.md"
    subprocess_log = run_dir / "subprocess.log"

    has_report = report.exists()
    report_status = _parse_report_status(report) if has_report else "missing"
    has_subprocess = subprocess_log.exists()

    last_rc, last_end_ts, last_cmd, last_err, tb = _parse_subprocess_log(subprocess_log) if has_subprocess else (None, "missing", "missing", "missing", False)

    params = _format_params_from_run_id(run_id)

    if has_subprocess:
        if last_rc is None:
            # Has log but no END line; if START exists this is incomplete.
            txt = _safe_read_text(subprocess_log, max_bytes=500_000)
            status = "INCOMPLETE" if "START " in txt and "END " not in txt else "UNKNOWN"
        else:
            if last_rc != 0 or tb:
                status = "FAIL"
            else:
                status = "SUCCESS"
    else:
        status = "SUCCESS" if has_report else "UNKNOWN"

    # Prefer a meaningful error label
    error = "—"
    if status in {"FAIL", "INCOMPLETE"}:
        error = last_err if last_err != "missing" else ("Traceback" if tb else "missing")
    elif status == "UNKNOWN":
        error = "no subprocess.log and no REPORT.md"

    return RunAudit(
        rel_path=rel,
        run_id=run_id,
        has_report=has_report,
        report_status=report_status,
        has_subprocess_log=has_subprocess,
        last_rc=last_rc,
        last_end_ts=last_end_ts,
        last_cmd=last_cmd,
        status=status,
        error=error,
        params=params,
    )


def _discover_run_dirs(runs_root: Path, include_failed: bool) -> list[Path]:
    if not runs_root.exists():
        return []

    candidates: set[Path] = set()
    for pat in ("**/subprocess.log", "**/REPORT.md", "**/run.log"):
        for p in runs_root.glob(pat):
            if not p.is_file():
                continue
            d = p.parent
            if not include_failed and "_failed" in d.parts:
                continue
            candidates.add(d)

    # Filter out known non-run utility dirs (like sweep rows/)
    out: list[Path] = []
    for d in sorted(candidates):
        if d.name in {"rows"}:
            continue
        out.append(d)
    return out


def _write_csv(rows: Iterable[RunAudit], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "rel_path",
                "run_id",
                "status",
                "last_rc",
                "last_end_ts",
                "has_report",
                "report_status",
                "has_subprocess_log",
                "error",
                "params",
                "last_cmd",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.rel_path,
                    r.run_id,
                    r.status,
                    "" if r.last_rc is None else str(r.last_rc),
                    r.last_end_ts,
                    "1" if r.has_report else "0",
                    r.report_status,
                    "1" if r.has_subprocess_log else "0",
                    r.error,
                    r.params,
                    r.last_cmd,
                ]
            )


def _render_markdown(rows: list[RunAudit]) -> str:
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.status] = counts.get(r.status, 0) + 1

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = [
        f"# TMCMC runs audit ({now})",
        "",
        "## Summary",
        "",
        "| status | count |",
        "|---|---:|",
    ]
    for k in sorted(counts.keys()):
        header.append(f"| {k} | {counts[k]} |")
    header.append("")

    lines = header + [
        "## Runs",
        "",
        "| rel_path | status | rc | report | report_status | error | params |",
        "|---|---|---:|---|---|---|---|",
    ]
    for r in rows:
        rc = "—" if r.last_rc is None else str(r.last_rc)
        report = "yes" if r.has_report else "no"
        lines.append(
            f"| `{r.rel_path}` | **{r.status}** | {rc} | {report} | {r.report_status} | {r.error} | {r.params} |"
        )
    lines.append("")
    return "\n".join(lines)


def _move_failed(rows: list[RunAudit], runs_root: Path, failed_root: Path, apply: bool) -> tuple[int, int]:
    moved = 0
    skipped = 0
    failed_root.mkdir(parents=True, exist_ok=True)

    for r in rows:
        if r.status != "FAIL":
            continue
        src = runs_root / r.rel_path
        dst = failed_root / r.rel_path
        if not src.exists():
            skipped += 1
            continue
        if dst.exists():
            skipped += 1
            continue

        if not apply:
            print(f"[dry-run] mv '{src}' -> '{dst}'")
            moved += 1
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        moved += 1

    return moved, skipped


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--runs-root", type=str, default="tmcmc/_runs", help="Runs root directory")
    p.add_argument("--include-failed", action="store_true", default=False, help="Include already-isolated _failed runs in the audit")
    p.add_argument("--md-out", type=str, default=None, help="Write markdown report to this path (otherwise prints)")
    p.add_argument("--csv-out", type=str, default=None, help="Write CSV to this path")
    p.add_argument("--move-failed", action="store_true", default=False, help="Move FAIL runs under --failed-root")
    p.add_argument(
        "--move-status",
        type=str,
        default="FAIL",
        help="Comma-separated statuses to move when --move-failed is set (e.g. FAIL,INCOMPLETE)",
    )
    p.add_argument("--failed-root", type=str, default="tmcmc/_runs/_failed", help="Destination root for failed run isolation")
    p.add_argument("--apply", action="store_true", default=False, help="Actually move (otherwise dry-run)")
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    runs_root = Path(args.runs_root).resolve()
    failed_root = Path(args.failed_root).resolve()

    run_dirs = _discover_run_dirs(runs_root, include_failed=bool(args.include_failed))
    rows = [_classify(d, runs_root) for d in run_dirs]

    # Stable ordering: FAIL first, then INCOMPLETE, then UNKNOWN, then SUCCESS
    order = {"FAIL": 0, "INCOMPLETE": 1, "UNKNOWN": 2, "SUCCESS": 3}
    rows.sort(key=lambda r: (order.get(r.status, 9), r.rel_path))

    md = _render_markdown(rows)
    if args.md_out:
        out = Path(args.md_out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"Wrote markdown: {out}")
    else:
        print(md)

    if args.csv_out:
        _write_csv(rows, Path(args.csv_out).resolve())
        print(f"Wrote csv: {Path(args.csv_out).resolve()}")

    if args.move_failed:
        statuses = {s.strip().upper() for s in str(args.move_status).split(",") if s.strip()}
        moved = 0
        skipped = 0
        # Move each status by reusing the existing mover (kept small on purpose)
        for st in sorted(statuses):
            if st != "FAIL":
                # Temporarily treat selected status as FAIL for the mover
                tmp_rows = [RunAudit(**{**r.__dict__, "status": "FAIL"}) if r.status == st else r for r in rows]
                m, s = _move_failed(tmp_rows, runs_root=runs_root, failed_root=failed_root, apply=bool(args.apply))
            else:
                m, s = _move_failed(rows, runs_root=runs_root, failed_root=failed_root, apply=bool(args.apply))
            moved += m
            skipped += s
        print(
            f"move_failed: moved={moved} skipped={skipped} apply={bool(args.apply)} "
            f"dst={failed_root} statuses={sorted(statuses)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

