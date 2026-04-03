from __future__ import annotations

import argparse

from nife.comets.run import main as _run_main


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", type=str, default="comets_runs")
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--max-cycles", type=int, default=250)
    p.add_argument("--time-step", type=float, default=0.02)
    p.add_argument("--glucose", type=float, default=0.11)
    p.add_argument("--anaerobic", action="store_true")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--progress", action="store_true")
    p.add_argument("--delete-files", action="store_true")
    p.add_argument("--save-config", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    forwarded = [
        "--output-root",
        args.output_root,
        "--run-name",
        args.run_name,
        "--max-cycles",
        str(args.max_cycles),
        "--time-step",
        str(args.time_step),
        "--threads",
        str(args.threads),
        "--met",
        f"glc__D_e={args.glucose}",
        "--write-media-log",
    ]
    if args.anaerobic:
        forwarded.append("--anaerobic")
    if args.progress:
        forwarded.append("--progress")
    if args.delete_files:
        forwarded.append("--delete-files")
    if args.save_config:
        forwarded.append("--save-config")
    return _run_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
