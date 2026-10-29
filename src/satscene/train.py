"""Training entry point: ``python -m satscene.train --config configs/default.yaml``.

For a minibatch of B logits z_i and class labels y_i, the default objective is
mean multiclass cross-entropy:
L = -(1/B) sum_i log softmax(z_i)[y_i]. Optional class weighting changes
each sample's contribution through its target-class weight. Validation scores
select the epoch to checkpoint; they do not estimate unseen-test performance.
See docs/MATHEMATICAL_METHODS.md for the complete objective and optimizer notes.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch import optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, f1_score

from .data import make_loader
from .labels import class_distribution, collect_split, inverse_frequency_weights, read_class_names
from .model import build_model, count_params_m, save_checkpoint
from .runtime import best_device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_loader(model, loader, device, criterion):
    model.eval()
    loss_sum = total = 0
    targets, predictions = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss_sum += criterion(logits, labels).item() * labels.size(0)
        targets.extend(labels.cpu().tolist())
        predictions.extend(logits.argmax(1).cpu().tolist())
        total += labels.size(0)
    if not total:
        raise ValueError("Validation loader is empty")
    return {
        "val_loss": loss_sum / total,
        "val_acc": 100.0 * sum(t == p for t, p in zip(targets, predictions)) / total,
        "val_macro_f1": f1_score(targets, predictions, average="macro", zero_division=0),
        "val_balanced_acc": balanced_accuracy_score(targets, predictions),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    set_seed(cfg["seed"])
    device = best_device()
    use_amp = device.type == "cuda"
    root = cfg["data"]["root"]
    class_names = read_class_names(os.path.join(root, "data.yaml"))
    train_ix, val_ix = (collect_split(root, s) for s in ("train", cfg["data"].get("val_split", "valid")))
    if not train_ix.paths or not val_ix.paths:
        raise SystemExit(
            f"No labeled images found (train={len(train_ix.paths)}, val={len(val_ix.paths)}). "
            f"Check data.root={root!r}, split folders, and YOLO label files."
        )
    for name, ix in (("train", train_ix), ("val", val_ix)):
        if ix.skipped:
            print(f"[warn] {len(ix.skipped)} {name} images skipped (missing/empty label file)")
    print("Train class distribution:", class_distribution(train_ix.labels, class_names))

    tc, oc = cfg["train"], cfg["optim"]
    train_loader = make_loader(train_ix.paths, train_ix.labels, cfg["data"]["img_size"], tc["batch_size"], True, tc["num_workers"])
    val_loader = make_loader(val_ix.paths, val_ix.labels, cfg["data"]["img_size"], tc["batch_size"], False, tc["num_workers"])

    model = build_model(len(class_names), cfg["model"]["name"], cfg["model"]["pretrained"],
                        img_size=cfg["data"]["img_size"]).to(device)
    print(f"Model on {device}: {count_params_m(model):.2f}M parameters")

    weight = None
    if tc.get("class_weighted_loss", False):
        weight = torch.tensor(inverse_frequency_weights(train_ix.labels, len(class_names)), dtype=torch.float, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.AdamW(model.parameters(), lr=oc["lr"], weight_decay=oc["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tc["epochs"])
    scaler = GradScaler("cuda", enabled=use_amp)

    out_dir = cfg["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    selection_metric = tc.get("selection_metric", "val_macro_f1")
    if selection_metric not in {"val_acc", "val_macro_f1", "val_balanced_acc"}:
        raise ValueError("train.selection_metric must be val_acc, val_macro_f1, or val_balanced_acc")
    history, best_score, best_epoch, best_acc = [], float("-inf"), 0, 0.0
    t0 = time.time()
    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        loop = tqdm(train_loader, desc=f"epoch {epoch}/{tc['epochs']}")
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
            loop.set_postfix(loss=f"{loss_sum/total:.4f}", acc=f"{100.*correct/total:.2f}")
        val_metrics = evaluate_loader(model, val_loader, device, criterion)
        row = {"epoch": epoch, "train_loss": loss_sum / total,
               "train_acc": 100.0 * correct / total,
               "lr": optimizer.param_groups[0]["lr"], **val_metrics}
        history.append(row)
        print(f"  val acc {row['val_acc']:.2f}% | macro-F1 {row['val_macro_f1']:.3f} "
              f"| val loss {row['val_loss']:.4f}")
        score = row[selection_metric]
        if score > best_score:
            best_score, best_acc, best_epoch = score, row["val_acc"], epoch
            save_checkpoint(model, os.path.join(out_dir, "best_model.pth"), class_names,
                            cfg["model"]["name"], cfg["data"]["img_size"])
        scheduler.step()

    with open(os.path.join(out_dir, "history.json"), "w") as fh:
        json.dump({"history": history, "best_val_acc": best_acc, "best_score": best_score,
                   "selection_metric": selection_metric, "best_epoch": best_epoch,
                   "class_names": class_names, "seconds": time.time() - t0, "config": cfg}, fh, indent=2)
    print(f"Best {selection_metric} {best_score:.4f} at epoch {best_epoch} (val acc {best_acc:.2f}%)")


if __name__ == "__main__":
    main()
