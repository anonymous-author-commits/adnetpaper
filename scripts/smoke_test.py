#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


REQUIRED = [
    "config_resolved.yaml",
    "metadata.json",
    "logs.txt",
    "metrics.json",
    "confusion_matrix.csv",
    "per_class_metrics.csv",
]


def latest_run(outputs_dir: Path) -> Path | None:
    runs = [d for d in outputs_dir.iterdir() if d.is_dir() and d.name != "tables"]
    if not runs:
        return None
    return sorted(runs, key=lambda p: p.stat().st_mtime)[-1]


def main():
    p = argparse.ArgumentParser(description="Dry-run smoke test for all configs.")
    p.add_argument("--configs-dir", default="configs")
    p.add_argument("--python", default=sys.executable)
    args = p.parse_args()

    cfgs = sorted(Path(args.configs_dir).glob("*.yaml"))
    if not cfgs:
        raise SystemExit("No configs found.")

    for cfg in cfgs:
        with cfg.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if str(data.get("mode", "")).lower() == "gradcam":
            continue
        cmd = [args.python, "scripts/run.py", "--mode", "dry-run", "--config", str(cfg)]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
        run = latest_run(Path("outputs"))
        if run is None:
            raise SystemExit("No run output folder found.")
        missing = [name for name in REQUIRED if not (run / name).exists()]
        if missing:
            raise SystemExit(f"Smoke test failed for {cfg}: missing {missing}")
    print("Smoke test passed.")


if __name__ == "__main__":
    main()

