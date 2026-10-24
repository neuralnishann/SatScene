"""Held-out evaluation:  ``python -m satscene.evaluate --config configs/default.yaml --split test``

Writes ``<output.dir>/metrics_<split>.json`` and ``confusion_<split>.png``.
Run this ONCE on the test split after model selection is finished.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix

from .data import make_loader
from .labels import collect_split, read_class_names
from .metrics import wilson_interval
from .model import build_model, load_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root, out_dir = cfg["data"]["root"], cfg["output"]["dir"]
    class_names = read_class_names(os.path.join(root, "data.yaml"))
    ix = collect_split(root, args.split)
    loader = make_loader(ix.paths, ix.labels, cfg["data"]["img_size"], cfg["train"]["batch_size"], False, cfg["train"]["num_workers"])

    model = build_model(len(class_names), cfg["model"]["name"], pretrained=False)
    model = load_checkpoint(model, args.checkpoint or os.path.join(out_dir, "best_model.pth"), device)

    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            y_pred += model(imgs.to(device)).argmax(1).cpu().tolist()
            y_true += labels.tolist()
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    k, n = int((y_true == y_pred).sum()), len(y_true)
    lo, hi = wilson_interval(k, n)
    rep = classification_report(y_true, y_pred, labels=list(range(len(class_names))),
                                target_names=class_names, output_dict=True, zero_division=0)
    metrics = {"split": args.split, "n": n, "accuracy": k / n, "accuracy_ci95": [lo, hi],
               "macro_f1": rep["macro avg"]["f1-score"], "per_class": {c: rep[c] for c in class_names}}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"metrics_{args.split}.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax, cbar=False)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"Confusion matrix ({args.split}, n={n})")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"confusion_{args.split}.png"), dpi=200)
    print(f"{args.split}: acc={k/n:.4f} (95% CI {lo:.3f}-{hi:.3f}), macro-F1={metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
