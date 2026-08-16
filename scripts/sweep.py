#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser(description="Run a list of configs, then aggregate tables.")
    p.add_argument("--configs", nargs="+", required=True)
    p.add_argument("--mode", default="train", choices=["train", "dry-run", "eval"])
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--aggregate-only", action="store_true")
    args = p.parse_args()

    if not args.aggregate_only:
        for cfg in args.configs:
            run_cmd([args.python, "scripts/run.py", "--mode", args.mode, "--config", cfg])
    run_cmd([args.python, "scripts/aggregate_results.py", "--outputs-dir", "outputs"])
    print(f"Done. Tables in {Path('outputs') / 'tables'}")


if __name__ == "__main__":
    main()

