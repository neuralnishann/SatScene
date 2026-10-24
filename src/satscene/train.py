"""Training entry point:  ``python -m satscene.train --config configs/default.yaml``"""
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

from .data import make_loader
from .labels import class_distribution, collect_split, inverse_frequency_weights, read_class_names
from .model import build_model, count_params_m


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_loader(model, loader, device):
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        correct += (model(imgs).argmax(1) == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    root = cfg["data"]["root"]
    class_names = read_class_names(os.path.join(root, "data.yaml"))
    train_ix, val_ix = (collect_split(root, s) for s in ("train", cfg["data"].get("val_split", "valid")))
    for name, ix in (("train", train_ix), ("val", val_ix)):
        if ix.skipped:
            print(f"[warn] {len(ix.skipped)} {name} images skipped (missing/empty label file)")
    print("Train class distribution:", class_distribution(train_ix.labels, class_names))

    tc, oc = cfg["train"], cfg["optim"]
    train_loader = make_loader(train_ix.paths, train_ix.labels, cfg["data"]["img_size"], tc["batch_size"], True, tc["num_workers"])
    val_loader = make_loader(val_ix.paths, val_ix.labels, cfg["data"]["img_size"], tc["batch_size"], False, tc["num_workers"])

    model = build_model(len(class_names), cfg["model"]["name"], cfg["model"]["pretrained"]).to(device)
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
    history, best_acc, best_epoch = [], 0.0, 0
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
        scheduler.step()
        val_acc = evaluate_loader(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": loss_sum / total, "train_acc": 100.0 * correct / total, "val_acc": val_acc})
        print(f"  val acc {val_acc:.2f}%")
        if val_acc > best_acc:  # strict '>' as in the original notebook
            best_acc, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pth"))

    with open(os.path.join(out_dir, "history.json"), "w") as fh:
        json.dump({"history": history, "best_val_acc": best_acc, "best_epoch": best_epoch,
                   "class_names": class_names, "seconds": time.time() - t0, "config": cfg}, fh, indent=2)
    print(f"Best val acc {best_acc:.2f}% at epoch {best_epoch}")


if __name__ == "__main__":
    main()
