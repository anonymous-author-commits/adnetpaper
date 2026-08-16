from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    num_classes: int,
    compute_roc_auc: bool = True,
) -> dict[str, Any]:
    p, r, f, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        zero_division=0,
    )
    out: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "per_class": [
            {
                "class_index": i,
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(support[i]),
            }
            for i in range(num_classes)
        ],
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(num_classes))
        ).tolist(),
        "roc_auc_ovr_macro": None,
    }
    if compute_roc_auc and y_prob is not None:
        try:
            y_true_bin = np.eye(num_classes)[y_true]
            out["roc_auc_ovr_macro"] = float(
                roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")
            )
        except Exception:
            out["roc_auc_ovr_macro"] = None
    return out


def write_confusion_matrix_csv(path: Path, cm: list[list[int]], class_names: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred"] + class_names)
        for i, row in enumerate(cm):
            writer.writerow([class_names[i]] + row)


def write_per_class_csv(path: Path, per_class: list[dict[str, Any]], class_names: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_index", "class_name", "precision", "recall", "f1", "support"])
        for row in per_class:
            i = row["class_index"]
            writer.writerow(
                [i, class_names[i], row["precision"], row["recall"], row["f1"], row["support"]]
            )


def write_blank_metric_files(
    metrics_json: Path, cm_csv: Path, per_class_csv: Path, class_names: list[str]
) -> None:
    import json

    with metrics_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy": None,
                "macro_f1": None,
                "weighted_f1": None,
                "roc_auc_ovr_macro": None,
                "note": "dry-run placeholder",
            },
            f,
            indent=2,
        )
    write_confusion_matrix_csv(cm_csv, [[0 for _ in class_names] for _ in class_names], class_names)
    write_per_class_csv(
        per_class_csv,
        [
            {"class_index": i, "precision": None, "recall": None, "f1": None, "support": 0}
            for i in range(len(class_names))
        ],
        class_names,
    )

