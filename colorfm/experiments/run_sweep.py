from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colorfm.train.pretrain import train_from_config
from colorfm.utils.experiment import apply_overrides, load_yaml, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small research sweep from a YAML manifest.")
    parser.add_argument("--sweep", type=Path, required=True)
    args = parser.parse_args()
    sweep = load_yaml(args.sweep)
    results = []
    for exp in sweep.get("experiments", []):
        config_path = Path(exp.get("config", sweep.get("base_config")))
        cfg = load_yaml(config_path)
        cfg = apply_overrides(cfg, exp.get("overrides"))
        cfg["run_name"] = exp.get("name", cfg.get("run_name", config_path.stem))
        print(f"\n=== running {cfg['run_name']} from {config_path} ===")
        result = train_from_config(cfg, config_path)
        results.append(result)
    out_path = Path("results") / "tables" / f"{sweep.get('name', 'sweep')}_train_results.json"
    save_json({"sweep": sweep.get("name"), "results": results}, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
