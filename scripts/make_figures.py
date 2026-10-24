"""Regenerate result figures from results/training_log.json.

    python scripts/make_figures.py            # writes results/figures/*.png and paper/figures/*.pdf
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from satscene.metrics import wilson_interval  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150})

with open(os.path.join(ROOT, "results", "training_log.json")) as fh:
    log = json.load(fh)

h = log["history"]
ep = [r["epoch"] for r in h]
best_ep, best = log["best_epoch"], log["best_val_acc"]
n_val = log["split_sizes"]["valid"]
lo, hi = wilson_interval(round(best / 100 * n_val), n_val)


def save(fig, name):
    for d, ext in (("results/figures", "png"), ("paper/figures", "pdf")):
        os.makedirs(os.path.join(ROOT, d), exist_ok=True)
        fig.savefig(os.path.join(ROOT, d, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


# --- Fig 1: training curves ---------------------------------------------------
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.8))
a.plot(ep, [r["train_loss"] for r in h], "o-", color="#1f77b4")
a.set_xlabel("Epoch"); a.set_ylabel("Training loss"); a.set_title("(a) Training loss")
b.plot(ep, [r["train_acc"] for r in h], "o-", label="Train (augmented)", color="#1f77b4")
b.plot(ep, [r["val_acc"] for r in h], "s-", label="Validation", color="#d62728")
b.axvline(best_ep, ls="--", color="gray", lw=1)
b.annotate(f"best ckpt\nepoch {best_ep}: {best:.1f}%", (best_ep, best), (best_ep - 5.3, 57),
           arrowprops=dict(arrowstyle="->", color="gray"), fontsize=8)
b.set_xlabel("Epoch"); b.set_ylabel("Accuracy (%)"); b.set_title("(b) Accuracy"); b.legend(loc="lower right", fontsize=8)
b.set_ylim(45, 102)
fig.tight_layout()
save(fig, "training_curves")

# --- Fig 2: class distribution ------------------------------------------------
dist = log["train_class_distribution"]
names = list(dist)  # already sorted descending
vals = [dist[k] for k in names]
fig, ax = plt.subplots(figsize=(7.2, 2.8))
ax.bar(names, vals, color="#4c72b0")
for i, v in enumerate(vals):
    ax.text(i, v + 1.5, str(v), ha="center", fontsize=8)
ax.set_ylabel("Training images"); ax.set_title(f"Training-set class distribution (N={sum(vals)}, imbalance {max(vals)/min(vals):.1f}:1)")
plt.setp(ax.get_xticklabels(), rotation=40, ha="right"); ax.grid(axis="x", visible=False)
fig.tight_layout()
save(fig, "class_distribution")

print(f"best val acc {best:.1f}% (n={n_val}); Wilson 95% CI = [{lo*100:.1f}, {hi*100:.1f}]")
print(f"final train-val gap = {h[-1]['train_acc'] - h[-1]['val_acc']:.1f} pp")
