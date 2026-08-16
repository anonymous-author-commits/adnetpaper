from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from adnet.data import build_dataloaders_from_config, class_weights_from_samples
from adnet.losses import build_loss
from adnet.metrics import compute_metrics
from adnet.models import build_model


def _run_loader(
    model: nn.Module,
    loader,
    device: torch.device,
    criterion: nn.Module | None = None,
    optimizer: optim.Optimizer | None = None,
    max_batches: int | None = None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    is_train = optimizer is not None and criterion is not None
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    n_samples = 0
    y_true, y_pred, y_prob = [], [], []

    ctx = torch.enable_grad if is_train else torch.no_grad
    with ctx():
        for bidx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            if criterion is not None:
                loss = criterion(logits, y)
                total_loss += float(loss.item()) * x.size(0)
                if is_train:
                    loss.backward()
                    optimizer.step()

            y_true.append(y.detach().cpu().numpy())
            y_pred.append(preds.detach().cpu().numpy())
            y_prob.append(probs.detach().cpu().numpy())
            n_samples += x.size(0)
            if max_batches is not None and bidx + 1 >= max_batches:
                break

    mean_loss = total_loss / max(1, n_samples)
    return (
        mean_loss,
        np.concatenate(y_true, axis=0),
        np.concatenate(y_pred, axis=0),
        np.concatenate(y_prob, axis=0),
    )


def train_and_evaluate(
    cfg: dict[str, Any],
    run_dir: Path,
    logger,
    mode: str,
    checkpoint_path: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    device = torch.device(cfg["device"] if cfg.get("device") else ("cuda" if torch.cuda.is_available() else "cpu"))
    loaders, class_to_idx, data_meta = build_dataloaders_from_config(cfg)
    num_classes = len(class_to_idx)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(num_classes)]

    model = build_model(cfg["model"], num_classes=num_classes).to(device)
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        logger.info("Loaded checkpoint: %s", checkpoint_path)

    train_samples = data_meta.pop("train_samples")
    use_class_weights = bool(cfg["loss"].get("use_class_weights", False))
    class_weights = None
    if use_class_weights:
        class_weights = class_weights_from_samples(train_samples, num_classes).to(device)
        logger.info("Class weights enabled: %s", class_weights.detach().cpu().tolist())
    else:
        logger.info("Class weights disabled.")

    criterion = build_loss(cfg["loss"], class_weights=class_weights)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
    )

    train_history = []
    if mode == "train":
        best_metric = -1.0
        best_state = None
        patience = int(cfg["training"].get("patience", 5))
        bad_epochs = 0
        epochs = int(cfg["training"]["epochs"])
        for epoch in range(1, epochs + 1):
            tr_loss, _, _, _ = _run_loader(
                model,
                loaders["train"],
                device,
                criterion=criterion,
                optimizer=optimizer,
            )
            _, yv, pv, qv = _run_loader(model, loaders["val"], device, criterion=None, optimizer=None)
            val_metrics = compute_metrics(
                yv,
                pv,
                qv,
                num_classes=num_classes,
                compute_roc_auc=bool(cfg["evaluation"].get("compute_roc_auc", True)),
            )
            score = val_metrics["macro_f1"]
            train_history.append({"epoch": epoch, "train_loss": tr_loss, "val_macro_f1": score})
            logger.info(
                "Epoch %03d | train_loss=%.4f | val_acc=%.4f | val_macro_f1=%.4f",
                epoch,
                tr_loss,
                val_metrics["accuracy"],
                score,
            )
            if score > best_metric + 1e-8:
                best_metric = score
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break
        if best_state is not None:
            ckpt = run_dir / "best_model.pt"
            torch.save(best_state, ckpt)
            model.load_state_dict(best_state)
            logger.info("Saved checkpoint: %s", ckpt)

    max_batches = 1 if mode == "dry-run" else None
    _, yt, yp, yprob = _run_loader(
        model,
        loaders[cfg["evaluation"].get("split", "test")],
        device,
        criterion=None,
        optimizer=None,
        max_batches=max_batches,
    )

    metrics = compute_metrics(
        yt,
        yp,
        yprob,
        num_classes=num_classes,
        compute_roc_auc=bool(cfg["evaluation"].get("compute_roc_auc", True)),
    )
    metrics["mode"] = mode
    metrics["run_dir"] = str(run_dir)
    metrics["class_names"] = class_names
    metrics["class_weights"] = (
        class_weights.detach().cpu().tolist() if class_weights is not None else None
    )
    if mode == "dry-run":
        metrics["note"] = "Dry-run executes one eval batch only; numbers are not reportable."
    if train_history:
        metrics["train_history"] = train_history
    metadata_patch = {"data": data_meta, "device": str(device)}
    return metrics, metadata_patch


def predict(
    cfg: dict[str, Any],
    checkpoint_path: str,
    logger,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    device = torch.device(cfg["device"] if cfg.get("device") else ("cuda" if torch.cuda.is_available() else "cpu"))
    loaders, class_to_idx, data_meta = build_dataloaders_from_config(cfg)
    data_meta.pop("train_samples", None)
    num_classes = len(class_to_idx)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    model = build_model(cfg["model"], num_classes=num_classes).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    logger.info("Loaded checkpoint for prediction: %s", checkpoint_path)

    rows: list[dict[str, Any]] = []
    split = cfg["evaluation"].get("split", "test")
    with torch.no_grad():
        offset = 0
        for x, y in loaders[split]:
            x = x.to(device)
            probs = torch.softmax(model(x), dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            ys = y.numpy()
            for i in range(len(preds)):
                rows.append(
                    {
                        "index": offset + i,
                        "true_index": int(ys[i]),
                        "true_label": idx_to_class[int(ys[i])],
                        "pred_index": int(preds[i]),
                        "pred_label": idx_to_class[int(preds[i])],
                        "max_prob": float(probs[i, preds[i]]),
                    }
                )
            offset += len(preds)
    return rows, {"data": data_meta, "device": str(device)}
