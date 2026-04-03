from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _detect_comets_home(explicit: str | None) -> str:
    candidates = [
        explicit,
        os.environ.get("COMETS_HOME"),
        "/opt/comets_linux",
        "/home/nishioka/comets_linux",
    ]
    for p in candidates:
        if not p:
            continue
        pp = Path(p).expanduser()
        if (pp / "bin").exists() and (pp / "lib").exists():
            return str(pp)
    raise FileNotFoundError(
        "COMETS_HOME が見つかりません。COMETS本体の bin/ と lib/ があるディレクトリを指す必要があります。"
    )


def _parse_met_kv(items: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--met は metabolite=value 形式です: {item}")
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            raise ValueError(f"--met の metabolite が空です: {item}")
        out[k] = float(v)
    return out


def _load_cobra_model(spec: str):
    import cobra

    p = Path(spec)
    if p.exists():
        suffix = p.suffix.lower()
        if suffix in {".xml", ".sbml"}:
            return cobra.io.read_sbml_model(str(p))
        if suffix == ".json":
            return cobra.io.load_json_model(str(p))
        if suffix in {".yml", ".yaml"}:
            return cobra.io.load_yaml_model(str(p))
        return cobra.io.load_model(str(p))
    return cobra.io.load_model(spec)


def _parse_grid(value: str) -> list[int]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 2:
        raise ValueError("--grid は 'X,Y' 形式です（例: 1,1）")
    return [int(parts[0]), int(parts[1])]


def _default_media(glucose: float, anaerobic: bool, extra: dict[str, float]) -> dict[str, float]:
    media = {
        "glc__D_e": float(glucose),
        "o2_e": 0.0 if anaerobic else 1000.0,
        "nh4_e": 1000.0,
        "pi_e": 1000.0,
        "h2o_e": 1000.0,
        "h_e": 1000.0,
    }
    media.update(extra)
    return media


def _ensure_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _ensure_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_ensure_jsonable(v) for v in value]
    return value


@dataclass(frozen=True)
class RunSummary:
    comets_home: str
    output_root: str
    run_name: str
    run_dir: str
    total_biomass_csv: str
    media_csv: str | None



def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    p.add_argument("--comets-home", type=str, default="")
    p.add_argument("--output-root", type=Path, default=Path("comets_runs"))
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--save-config", action="store_true")

    p.add_argument("--cobra-model", type=str, default="textbook")
    p.add_argument("--initial-biomass", type=float, default=5e-6)
    p.add_argument("--optimizer", type=str, default="GLOP")
    p.add_argument("--obj-style", type=str, default="MAX_OBJECTIVE_MIN_TOTAL")

    p.add_argument("--grid", type=str, default="1,1")
    p.add_argument("--met", action="append", default=[])
    p.add_argument("--anaerobic", action="store_true")

    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--max-cycles", type=int, default=250)
    p.add_argument("--time-step", type=float, default=0.02)
    p.add_argument("--space-width", type=float, default=1.0)
    p.add_argument("--default-vmax", type=float, default=18.0)
    p.add_argument("--default-km", type=float, default=0.000003)
    p.add_argument("--write-media-log", action="store_true")
    p.add_argument("--death-rate", type=float, default=0.0)

    p.add_argument("--progress", action="store_true")
    p.add_argument("--delete-files", action="store_true")
    return p

def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    comets_home = _detect_comets_home(args.comets_home or None)
    os.environ["COMETS_HOME"] = comets_home

    import cobra
    import cometspy as c

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    os.chdir(output_root)

    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"run_{run_stamp}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    layout = c.layout()
    layout.grid = _parse_grid(args.grid)

    media = _default_media(
        glucose=0.11,
        anaerobic=bool(args.anaerobic),
        extra=_parse_met_kv(list(args.met)),
    )
    for k, v in sorted(media.items()):
        layout.set_specific_metabolite(k, float(v))

    cobra_model = _load_cobra_model(args.cobra_model)
    if not isinstance(cobra_model, cobra.Model):
        raise TypeError(f"Unexpected model type: {type(cobra_model)}")

    model = c.model(cobra_model)
    model.initial_pop = [0, 0, float(args.initial_biomass)]
    model.obj_style = str(args.obj_style)
    model.change_optimizer(str(args.optimizer))
    model.change_bounds("EX_glc__D_e", -1000, 1000)
    model.change_bounds("EX_ac_e", -1000, 1000)
    model.change_bounds("ATPM", 8, 1000)

    layout.add_model(model)

    params = c.params()
    params.set_param("numRunThreads", int(args.threads))
    params.set_param("defaultVmax", float(args.default_vmax))
    params.set_param("defaultKm", float(args.default_km))
    params.set_param("maxCycles", int(args.max_cycles))
    params.set_param("timeStep", float(args.time_step))
    params.set_param("spaceWidth", float(args.space_width))
    params.set_param("maxSpaceBiomass", 10)
    params.set_param("minSpaceBiomass", 1e-11)
    params.set_param("writeMediaLog", bool(args.write_media_log))
    params.set_param("deathRate", float(args.death_rate))

    exp = c.comets(layout, params, relative_dir=str(Path(run_name)) + "/")
    exp.run(delete_files=bool(args.delete_files), progress=bool(args.progress))

    total_biomass_csv = run_dir / "total_biomass.csv"
    exp.total_biomass.to_csv(total_biomass_csv, index=False)

    media_csv: Path | None = None
    if bool(args.write_media_log):
        media_csv = run_dir / "media.csv"
        exp.media.to_csv(media_csv, index=False)

    summary = RunSummary(
        comets_home=comets_home,
        output_root=str(output_root),
        run_name=str(run_name),
        run_dir=str(run_dir.resolve()),
        total_biomass_csv=str(total_biomass_csv.resolve()),
        media_csv=str(media_csv.resolve()) if media_csv else None,
    )

    if args.save_config:
        config = {
            "args": _ensure_jsonable(vars(args)),
            "summary": asdict(summary),
            "time": {"run_stamp": run_stamp},
        }
        (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))

    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
