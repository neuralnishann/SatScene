"""Plot train/validation curves from a packaged training run.

Usage: python scripts/plot_training_history.py runs/convnext_tiny/history.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("history", nargs="?", default="runs/convnext_tiny/history.json")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    source = Path(args.history)
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = payload["history"]
    if not rows or "val_loss" not in rows[0]:
        raise SystemExit("This history has no validation-loss curves. Re-run training with the current code.")
    epochs = [r["epoch"] for r in rows]
    best_epoch = payload.get("best_epoch")
    out = Path(args.output) if args.output else source.with_name("training_curves.png")

    fig, axes = plt.subplots(3, 2, figsize=(12, 11))
    panels = [
        ("train_loss", "val_loss", "Loss", "Cross-entropy"),
        ("train_acc", "val_acc", "Accuracy", "Percent"),
        (None, "val_macro_f1", "Macro F1", "Score"),
        (None, "val_balanced_acc", "Balanced accuracy", "Score"),
    ]
    for ax, (train_key, val_key, title, ylabel) in zip(axes.flat, panels):
        if train_key:
            ax.plot(epochs, [r[train_key] for r in rows], "o-", label="Train")
        ax.plot(epochs, [r[val_key] for r in rows], "s-", label="Validation")
        if best_epoch is not None:
            ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1, label="Selected checkpoint")
        ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
        ax.grid(alpha=0.25)
        ax.legend()
    if "lr" in rows[0]:
        axes[2, 0].plot(epochs, [r["lr"] for r in rows], "o-", color="#7A5195")
        axes[2, 0].set(title="Learning rate", xlabel="Epoch", ylabel="LR")
        axes[2, 0].grid(alpha=0.25)
    axes[2, 1].axis("off")
    fig.suptitle(f"Training history · selected by {payload.get('selection_metric', 'validation metric')}")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
