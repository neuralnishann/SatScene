"""Gradio demo:  ``python -m satscene.app --config configs/default.yaml``"""
from __future__ import annotations

import argparse
import os

import gradio as gr
import torch
import yaml

from .agent import SatelliteAgent
from .labels import read_class_names
from .model import build_model, load_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--share", action="store_true", help="create a temporary public Gradio link")
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = read_class_names(os.path.join(cfg["data"]["root"], "data.yaml"))
    model = build_model(len(names), cfg["model"]["name"], pretrained=False)
    model = load_checkpoint(model, args.checkpoint or os.path.join(cfg["output"]["dir"], "best_model.pth"), device)
    agent = SatelliteAgent(model, names, device, cfg["data"]["img_size"])

    def run(path):
        return "Please upload an image." if path is None else agent.analyze(path)

    gr.Interface(
        fn=run,
        inputs=gr.Image(type="filepath", label="Satellite image"),
        outputs=gr.Markdown(label="Agent report"),
        title="Satellite Scene Agent",
        description="Upload an overhead image: the model predicts the land-use class and the agent suggests follow-up analyses.",
    ).launch(share=args.share)


if __name__ == "__main__":
    main()
