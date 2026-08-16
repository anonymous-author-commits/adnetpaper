#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml


def _load_metrics(run_dir: Path) -> dict[str, Any] | None:
    mp = run_dir / "metrics.json"
    cp = run_dir / "config_resolved.yaml"
    if not mp.exists() or not cp.exists():
        return None
    with mp.open("r", encoding="utf-8") as f:
        m = json.load(f)
    with cp.open("r", encoding="utf-8") as f:
        c = yaml.safe_load(f) or {}
    return {"run_id": run_dir.name, "metrics": m, "config": c}


def _metric_row(item: dict[str, Any]) -> dict[str, Any]:
    m = item["metrics"]
    return {
        "run_id": item["run_id"],
        "accuracy": m.get("accuracy"),
        "macro_f1": m.get("macro_f1"),
        "weighted_f1": m.get("weighted_f1"),
        "roc_auc": m.get("roc_auc_ovr_macro"),
    }


def _pick_best(rows: list[dict[str, Any]], key: str = "macro_f1") -> dict[str, Any] | None:
    valid = [r for r in rows if isinstance(r.get(key), (int, float))]
    if not valid:
        return None
    return max(valid, key=lambda r: r[key])


def _write_table(path: Path, headers: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _blank_table(path: Path, row_names: list[str]) -> None:
    rows = [
        {"setting": name, "accuracy": "—", "macro_f1": "—", "weighted_f1": "—", "roc_auc": "—"}
        for name in row_names
    ]
    _write_table(path, ["setting", "accuracy", "macro_f1", "weighted_f1", "roc_auc"], rows)


def _tex_stub(csv_path: Path, tex_path: Path, caption: str, label: str) -> None:
    text = f"""% Auto-generated table stub
\\begin{{table}}[t]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
% Fill values from {csv_path.as_posix()}
\\begin{{tabular}}{{lcccc}}
\\hline
Setting & Accuracy & Macro-F1 & Weighted-F1 & ROC-AUC \\\\
\\hline
... & ... & ... & ... & ... \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
    tex_path.write_text(text, encoding="utf-8")


def aggregate(outputs_dir: Path) -> Path:
    tables_dir = outputs_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    runs = [d for d in outputs_dir.iterdir() if d.is_dir() and d.name != "tables"]
    items = [x for x in (_load_metrics(d) for d in runs) if x]

    arch_rows: dict[str, list[dict[str, Any]]] = {
        "ADNET dual-stream": [],
        "ResNet-50 baseline": [],
        "EfficientNet-B0 baseline": [],
    }
    imb_rows: dict[str, list[dict[str, Any]]] = {
        "no weights": [],
        "class weights": [],
        "focal": [],
    }
    aug_rows: dict[str, list[dict[str, Any]]] = {"none": [], "baseline aug": [], "light aug": []}

    for item in items:
        cfg = item["config"]
        row = _metric_row(item)
        mtype = cfg.get("model", {}).get("type")
        if mtype == "adnet_dualstream":
            arch_rows["ADNET dual-stream"].append(row)
        elif mtype == "resnet50":
            arch_rows["ResNet-50 baseline"].append(row)
        elif mtype == "efficientnet_b0":
            arch_rows["EfficientNet-B0 baseline"].append(row)

        use_weights = bool(cfg.get("loss", {}).get("use_class_weights", False))
        ltype = cfg.get("loss", {}).get("type", "cross_entropy")
        if ltype == "focal":
            imb_rows["focal"].append(row)
        elif use_weights:
            imb_rows["class weights"].append(row)
        else:
            imb_rows["no weights"].append(row)

        aug = cfg.get("data", {}).get("augmentation", "baseline")
        if aug == "none":
            aug_rows["none"].append(row)
        elif aug == "light":
            aug_rows["light aug"].append(row)
        else:
            aug_rows["baseline aug"].append(row)

    arch_csv = tables_dir / "architecture_comparison.csv"
    imb_csv = tables_dir / "imbalance_ablation.csv"
    aug_csv = tables_dir / "augmentation_ablation.csv"

    if not items:
        _blank_table(arch_csv, list(arch_rows.keys()))
        _blank_table(imb_csv, list(imb_rows.keys()))
        _blank_table(aug_csv, list(aug_rows.keys()))
    else:
        _write_table(
            arch_csv,
            ["setting", "accuracy", "macro_f1", "weighted_f1", "roc_auc", "run_id"],
            [
                {"setting": k, **(_pick_best(v) or {"accuracy": "—", "macro_f1": "—", "weighted_f1": "—", "roc_auc": "—", "run_id": ""})}
                for k, v in arch_rows.items()
            ],
        )
        _write_table(
            imb_csv,
            ["setting", "accuracy", "macro_f1", "weighted_f1", "roc_auc", "run_id"],
            [
                {"setting": k, **(_pick_best(v) or {"accuracy": "—", "macro_f1": "—", "weighted_f1": "—", "roc_auc": "—", "run_id": ""})}
                for k, v in imb_rows.items()
            ],
        )
        _write_table(
            aug_csv,
            ["setting", "accuracy", "macro_f1", "weighted_f1", "roc_auc", "run_id"],
            [
                {"setting": k, **(_pick_best(v) or {"accuracy": "—", "macro_f1": "—", "weighted_f1": "—", "roc_auc": "—", "run_id": ""})}
                for k, v in aug_rows.items()
            ],
        )

    _tex_stub(arch_csv, tables_dir / "architecture_comparison.tex", "Architecture comparison.", "tab:arch")
    _tex_stub(imb_csv, tables_dir / "imbalance_ablation.tex", "Imbalance ablation.", "tab:imb")
    _tex_stub(aug_csv, tables_dir / "augmentation_ablation.tex", "Augmentation ablation.", "tab:aug")

    readme = tables_dir / "README.md"
    readme.write_text(
        (
            "# Tables\n\n"
            "Tables are built from `outputs/*/metrics.json`.\n\n"
            "If values are `—`, run experiments first. Example sweep:\n\n"
            "```bash\n"
            "python scripts/sweep.py --configs configs/adnet_dualstream.yaml configs/baseline_resnet50.yaml configs/baseline_efficientnetb0.yaml\n"
            "```\n"
        ),
        encoding="utf-8",
    )
    return tables_dir


def main():
    p = argparse.ArgumentParser(description="Aggregate ADNET run outputs into paper-ready tables.")
    p.add_argument("--outputs-dir", default="outputs")
    args = p.parse_args()
    tdir = aggregate(Path(args.outputs_dir))
    print(f"Saved tables to {tdir}")


if __name__ == "__main__":
    main()

