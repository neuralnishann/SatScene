"""Held-out evaluation:  ``python -m satscene.evaluate --config configs/default.yaml --split test``

Writes ``<output.dir>/metrics_<split>.json`` and ``confusion_<split>.png``.
Prediction CSVs store image paths relative to the configured dataset root.
Run this ONCE on the test split after model selection is finished.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix

from .data import make_loader
from .labels import collect_split, read_class_names
from .metrics import calibration_summary, multiclass_brier_score, wilson_interval
from .model import build_model, checkpoint_metadata, load_checkpoint
from .runtime import best_device


def _dataset_relative_image_path(image_path: str, dataset_root: str) -> str:
    return Path(image_path).relative_to(Path(dataset_root)).as_posix()


def _evaluate_checkpoint(cfg: dict, split: str, checkpoint: str, out_dir: str, device) -> None:
    root = cfg["data"]["root"]
    metadata = checkpoint_metadata(checkpoint)
    dataset_names = read_class_names(os.path.join(root, "data.yaml"))
    class_names = metadata.get("class_names", dataset_names)
    if list(class_names) != dataset_names:
        raise ValueError("Checkpoint class order does not match the dataset data.yaml")
    model_name = metadata.get("model_name", cfg["model"]["name"])
    img_size = int(metadata.get("img_size", cfg["data"]["img_size"]))
    ix = collect_split(root, split)
    if not ix.paths:
        raise SystemExit(f"No labeled images found for split {split!r} under {root!r}.")
    loader = make_loader(ix.paths, ix.labels, img_size, cfg["train"]["batch_size"], False, cfg["train"]["num_workers"])

    model = build_model(len(class_names), model_name, pretrained=False, img_size=img_size)
    model = load_checkpoint(model, checkpoint, device)

    y_true, y_pred, y_prob = [], [], []
    test_loss_sum = 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for imgs, labels in loader:
            logits = model(imgs.to(device))
            test_loss_sum += criterion(logits.cpu(), labels).item()
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            y_prob.extend(probabilities.tolist())
            y_pred += probabilities.argmax(1).tolist()
            y_true += labels.tolist()
    y_true, y_pred, y_prob = np.array(y_true), np.array(y_pred), np.array(y_prob)

    k, n = int((y_true == y_pred).sum()), len(y_true)
    lo, hi = wilson_interval(k, n)
    rep = classification_report(y_true, y_pred, labels=list(range(len(class_names))),
                                target_names=class_names, output_dict=True, zero_division=0)
    top_k = min(3, len(class_names))
    topk_pred = np.argsort(y_prob, axis=1)[:, -top_k:]
    calibration = calibration_summary(y_prob, y_true)
    metrics = {
        "split": split, "n": n, "accuracy": k / n, "accuracy_ci95": [lo, hi],
        "cross_entropy_loss": test_loss_sum / n,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": rep["macro avg"]["f1-score"],
        "weighted_f1": rep["weighted avg"]["f1-score"],
        f"top_{top_k}_accuracy": float((topk_pred == y_true[:, None]).any(axis=1).mean()),
        "ece_top_label": calibration["ece"],
        "multiclass_brier_score": multiclass_brier_score(y_prob, y_true, len(class_names)),
        "per_class": {c: rep[c] for c in class_names},
        "calibration_bins": calibration["bins"],
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"metrics_{split}.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    with open(os.path.join(out_dir, f"predictions_{split}.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image", "true_class", "predicted_class", "confidence", "correct"])
        writer.writeheader()
        for path, true_id, pred_id, prob in zip(ix.paths, y_true, y_pred, y_prob):
            image_path = _dataset_relative_image_path(path, root)
            writer.writerow({"image": image_path, "true_class": class_names[true_id],
                             "predicted_class": class_names[pred_id],
                             "confidence": float(prob[pred_id]), "correct": bool(true_id == pred_id)})

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, axes = plt.subplots(2, 2, figsize=(17, 13))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names,
                yticklabels=class_names, ax=axes[0, 0], cbar=False)
    axes[0, 0].set_title("Confusion matrix · counts")
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                xticklabels=class_names, yticklabels=class_names, ax=axes[0, 1], cbar=False)
    axes[0, 1].set_title("Confusion matrix · row-normalized recall")
    for ax in axes[0]:
        ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
        ax.tick_params(axis="x", labelrotation=50); ax.tick_params(axis="y", labelrotation=0)
    f1s = [rep[c]["f1-score"] for c in class_names]
    axes[1, 0].barh(class_names, f1s, color="#4472C4")
    axes[1, 0].set_xlim(0, 1); axes[1, 0].set_xlabel("F1 score")
    axes[1, 0].set_title("Per-class F1 (show support in report)")
    bins = calibration["bins"]
    if bins:
        axes[1, 1].plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], "o-", label="Model")
    axes[1, 1].plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    axes[1, 1].set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Empirical accuracy",
                   title=f"Reliability diagram (ECE={calibration['ece']:.3f})")
    axes[1, 1].legend()
    fig.suptitle(f"Held-out {split} evaluation · n={n} · accuracy={k/n:.3f} · macro-F1={metrics['macro_f1']:.3f}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"confusion_{split}.png"), dpi=200)
    plt.close(fig)
    print(f"{model_name} ({checkpoint})")
    print(f"{split}: acc={k/n:.4f} (95% CI {lo:.3f}-{hi:.3f}), "
          f"balanced acc={metrics['balanced_accuracy']:.4f}, macro-F1={metrics['macro_f1']:.4f}, "
          f"top-{top_k} acc={metrics[f'top_{top_k}_accuracy']:.4f}, ECE={metrics['ece_top_label']:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    checkpoint_group = ap.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--checkpoint", default=None)
    checkpoint_group.add_argument(
        "--checkpoints",
        nargs=3,
        metavar=("MODEL_1", "MODEL_2", "MODEL_3"),
        help="evaluate exactly three checkpoints in one run",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="output directory for one checkpoint (defaults to config output.dir)",
    )
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    if args.checkpoints and args.output_dir:
        ap.error("--output-dir cannot be used with --checkpoints; each model writes beside its checkpoint")
    if args.checkpoints:
        checkpoints = args.checkpoints
        output_dirs = [str(Path(checkpoint).resolve().parent) for checkpoint in checkpoints]
        if len(set(output_dirs)) != len(output_dirs):
            ap.error("the three checkpoints must be in separate directories to avoid overwriting outputs")
    else:
        output_dir = args.output_dir or cfg["output"]["dir"]
        checkpoints = [args.checkpoint or os.path.join(output_dir, "best_model.pth")]
        output_dirs = [output_dir]

    device = best_device()
    for checkpoint, output_dir in zip(checkpoints, output_dirs):
        if not Path(checkpoint).is_file():
            raise SystemExit(f"Checkpoint not found: {checkpoint}")
        _evaluate_checkpoint(cfg, args.split, checkpoint, output_dir, device)


if __name__ == "__main__":
    main()
