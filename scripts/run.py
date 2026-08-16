#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adnet.config import apply_overrides, load_config
from adnet.metrics import (
    write_blank_metric_files,
    write_confusion_matrix_csv,
    write_per_class_csv,
)
from adnet.train_eval import predict, train_and_evaluate
from adnet.utils import (
    ensure_dir,
    get_git_commit,
    make_run_id,
    set_determinism,
    setup_logging,
    write_json,
    write_yaml,
)


def parse_args():
    p = argparse.ArgumentParser(description="Unified ADNET experiment runner")
    p.add_argument("--mode", required=True, choices=["train", "eval", "predict", "gradcam", "dry-run"])
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--set", dest="overrides", action="append", default=[], help="Override as key=value")
    p.add_argument("--run-id", default=None, help="Optional custom run id")
    p.add_argument("--checkpoint", default=None, help="Checkpoint path for eval/predict/gradcam")
    p.add_argument("--k", type=int, default=8, help="K examples for gradcam mode")
    return p.parse_args()


def _write_metrics_artifacts(run_dir: Path, metrics: dict) -> None:
    write_json(run_dir / "metrics.json", metrics)
    class_names = metrics["class_names"]
    write_confusion_matrix_csv(
        run_dir / "confusion_matrix.csv",
        metrics["confusion_matrix"],
        class_names,
    )
    write_per_class_csv(run_dir / "per_class_metrics.csv", metrics["per_class"], class_names)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.overrides)
    run_id = args.run_id or make_run_id(cfg)
    run_dir = ensure_dir(Path("outputs") / run_id)
    logger = setup_logging(run_dir)

    set_determinism(int(cfg["seed"]), deterministic=bool(cfg.get("deterministic", True)))
    write_yaml(run_dir / "config_resolved.yaml", cfg)
    metadata = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "seed": int(cfg["seed"]),
        "device": cfg.get("device", "auto"),
        "git_commit": get_git_commit(),
        "mode": args.mode,
    }

    if args.mode == "gradcam":
        from scripts.gradcam_panels import generate_gradcam_panels

        ckpt = args.checkpoint or cfg.get("checkpoint")
        if not ckpt:
            raise ValueError("gradcam mode requires --checkpoint or config.checkpoint")
        meta = generate_gradcam_panels(cfg=cfg, checkpoint_path=ckpt, run_dir=run_dir, k=args.k, logger=logger)
        metadata.update(meta)
        write_json(run_dir / "metadata.json", metadata)
        idx_to_class = metadata.get("data", {}).get("idx_to_class", {})
        class_names = [idx_to_class[k] for k in sorted(idx_to_class.keys())] if idx_to_class else [f"class_{i}" for i in range(5)]
        write_blank_metric_files(
            run_dir / "metrics.json",
            run_dir / "confusion_matrix.csv",
            run_dir / "per_class_metrics.csv",
            class_names=class_names,
        )
        logger.info("Grad-CAM artifacts saved under: %s", run_dir / "figures")
        return

    if args.mode == "predict":
        ckpt = args.checkpoint or cfg.get("checkpoint")
        if not ckpt:
            raise ValueError("predict mode requires --checkpoint or config.checkpoint")
        rows, meta_patch = predict(cfg, checkpoint_path=ckpt, logger=logger)
        with (run_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["index"])
            writer.writeheader()
            if rows:
                writer.writerows(rows)
        metadata.update(meta_patch)
        write_json(run_dir / "metadata.json", metadata)
        idx_to_class = metadata.get("data", {}).get("idx_to_class", {})
        class_names = [idx_to_class[k] for k in sorted(idx_to_class.keys())] if idx_to_class else [f"class_{i}" for i in range(5)]
        write_blank_metric_files(
            run_dir / "metrics.json",
            run_dir / "confusion_matrix.csv",
            run_dir / "per_class_metrics.csv",
            class_names=class_names,
        )
        logger.info("Predictions saved: %s", run_dir / "predictions.csv")
        return

    mode = args.mode
    ckpt = args.checkpoint or cfg.get("checkpoint")
    if mode == "eval" and not ckpt:
        raise ValueError("eval mode requires --checkpoint or config.checkpoint")
    if mode == "dry-run":
        mode = "dry-run"
    metrics, meta_patch = train_and_evaluate(cfg=cfg, run_dir=run_dir, logger=logger, mode=mode, checkpoint_path=ckpt)
    metadata.update(meta_patch)
    write_json(run_dir / "metadata.json", metadata)
    _write_metrics_artifacts(run_dir, metrics)
    logger.info("Run complete. Output: %s", run_dir)


if __name__ == "__main__":
    main()
