"""Perceive -> Reason -> Report wrapper around the classifier.

This is a *rule-based decision-support layer*, not an autonomous LLM agent:
  1. Perception : image -> softmax probabilities (ConvNeXt-Tiny)
  2. Reasoning  : predicted class -> static knowledge-table lookup
  3. Report     : markdown summary (+ structured dict for programmatic use)
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F

from .data import build_transforms, read_rgb
from .knowledge import lookup


class SatelliteAgent:
    def __init__(self, model, class_names: List[str], device, img_size: int = 224,
                 top_k: int = 3, low_confidence: float = 0.5):
        self.model = model.eval()
        self.class_names, self.device = class_names, device
        self.tf = build_transforms(img_size, train=False)
        self.top_k, self.low_confidence = top_k, low_confidence

    @torch.no_grad()
    def perceive(self, image_path: str) -> torch.Tensor:
        x = self.tf(image=read_rgb(image_path))["image"].unsqueeze(0).to(self.device)
        return F.softmax(self.model(x), dim=1)[0].cpu()

    def analyze_structured(self, image_path: str) -> Dict:
        probs = self.perceive(image_path)
        k = min(self.top_k, len(self.class_names))
        top = torch.topk(probs, k)
        pred = self.class_names[top.indices[0].item()]
        title, action = lookup(pred)
        return {
            "class": pred,
            "confidence": top.values[0].item(),
            "top_k": [(self.class_names[i], p) for p, i in zip(top.values.tolist(), top.indices.tolist())],
            "title": title,
            "suggested_follow_up": action,
            "low_confidence": top.values[0].item() < self.low_confidence,
        }

    def analyze(self, image_path: str) -> str:
        r = self.analyze_structured(image_path)
        alts = ", ".join(f"{c} ({p*100:.1f}%)" for c, p in r["top_k"])
        flag = "\n\n> Low confidence: treat this label as tentative." if r["low_confidence"] else ""
        return (
            f"### Satellite Agent Analysis Report\n"
            f"**Primary land use:** `{r['class']}`  \n"
            f"**Confidence:** `{r['confidence']*100:.2f}%`  \n"
            f"**Top-{len(r['top_k'])}:** {alts}\n\n"
            f"**{r['title']}.** Suggested follow-up: {r['suggested_follow_up']}{flag}"
        )
