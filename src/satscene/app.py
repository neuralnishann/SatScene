"""Local browser GUI: ``python -m satscene.app --config configs/default.yaml``."""
from __future__ import annotations

if __package__ in (None, ""):
    raise SystemExit(
        "Run the browser app from the project root with `python -m satscene.app`; "
        "do not run src/satscene/app.py directly."
    )

import argparse
import os
from pathlib import Path

import gradio as gr
import yaml

from .agent import SatelliteAgent
from .labels import read_class_names
from .model import build_model, checkpoint_metadata, load_checkpoint
from .runtime import best_device


APP_CSS = """
.gradio-container {
    max-width: 1240px !important;
    margin: 0 auto !important;
    background: #f4f7fb !important;
}
.hero {
    padding: 30px 34px;
    margin: 14px 0 22px;
    border-radius: 22px;
    color: #f8fbff;
    background: linear-gradient(120deg, #10243e 0%, #173d61 58%, #12677a 100%);
    box-shadow: 0 16px 36px rgba(20, 48, 75, .16);
}
.hero-kicker {
    margin: 0 0 8px;
    color: #a9dce4;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: .14em;
    text-transform: uppercase;
}
.hero h1 {
    margin: 0;
    color: #fff;
    font-size: clamp(30px, 5vw, 46px);
    line-height: 1.08;
}
.hero p {
    max-width: 700px;
    margin: 12px 0 0;
    color: #d5e3ef;
    font-size: 16px;
    line-height: 1.6;
}
.panel {
    padding: 22px !important;
    border: 1px solid #e2eaf2 !important;
    border-radius: 18px !important;
    background: #fff !important;
    box-shadow: 0 8px 24px rgba(19, 47, 75, .055);
}
.section-label {
    color: #58748f;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: .1em;
    text-transform: uppercase;
}
.helper-copy {
    color: #71849a;
    font-size: 13px;
    line-height: 1.55;
}
.status-card {
    min-height: 72px;
    padding: 14px 18px !important;
    border: 1px solid #dce8f1 !important;
    border-radius: 14px !important;
    background: #f7fafc !important;
}
.status-card h3 {
    margin: 0 0 5px;
    color: #17324e;
}
.status-card p {
    margin: 0;
    color: #61758a;
}
.result-card {
    padding: 18px !important;
    border: 1px solid #d9eaf1 !important;
    border-radius: 14px !important;
    background: #f1f8fb !important;
}
.disclaimer {
    padding: 14px 18px;
    border-left: 3px solid #8bbec8;
    border-radius: 8px;
    background: #edf4f8;
    color: #536a7f;
    font-size: 13px;
    line-height: 1.55;
}
@media (max-width: 700px) {
    .hero { padding: 24px 20px; border-radius: 16px; }
    .panel { padding: 16px !important; }
}
"""

APP_THEME = gr.themes.Soft().set(
    background_fill_primary_dark="#f4f7fb",
    background_fill_secondary_dark="#edf4f8",
    body_background_fill_dark="#f4f7fb",
    body_text_color_dark="#17324e",
    body_text_color_subdued_dark="#61758a",
    block_background_fill_dark="#ffffff",
    block_label_text_color_dark="#58748f",
    block_title_text_color_dark="#17324e",
    input_background_fill_dark="#ffffff",
    input_placeholder_color_dark="#71849a",
    button_secondary_background_fill_dark="#e8eef4",
    button_secondary_background_fill_hover_dark="#dce7ef",
    button_secondary_border_color_dark="#d2dee8",
    button_secondary_text_color_dark="#17324e",
)


def _class_names(cfg: dict, checkpoint: str) -> list[str]:
    """Resolve class order from checkpoint metadata, config, or dataset YAML."""
    metadata = checkpoint_metadata(checkpoint)
    names = metadata.get("class_names") or cfg.get("class_names")
    if not names:
        yaml_path = os.path.join(cfg["data"]["root"], "data.yaml")
        if os.path.isfile(yaml_path):
            names = read_class_names(yaml_path)
    if not names:
        raise RuntimeError(
            "Class names are unavailable. Train with this version of SatScene, "
            "set class_names in the config, or make data.yaml available."
        )
    return list(names)


def _analyze_image(agent: SatelliteAgent, image_path: str | None) -> tuple[str, dict[str, float], str]:
    if not image_path:
        return "Choose an image, then select **Analyze image**.", {}, ""
    try:
        result = agent.analyze_structured(image_path)
    except (OSError, ValueError, RuntimeError) as exc:
        return f"**Could not analyze this image:** {exc}", {}, ""

    if result["low_confidence"]:
        status = "### Review recommended\nThe model is uncertain; treat this prediction as tentative."
    else:
        status = "### Analysis complete\nReview the ranked scene predictions below."

    predictions = {name: float(score) for name, score in result["top_k"]}
    follow_up = (
        f"### Suggested follow-up: {result['title']}\n"
        f"{result['suggested_follow_up']}\n\n"
        "This is a recommendation for an analyst; SatScene does not run the analysis."
    )
    return status, predictions, follow_up


def _clear_results() -> tuple[None, str, None, str]:
    return (
        None,
        "### Ready to analyze\nUpload an image, then select **Analyze image**.",
        None,
        "",
    )


def create_interface(agent: SatelliteAgent):
    """Build the upload-and-predict interface without starting a web server."""
    with gr.Blocks(title="SatScene · Satellite Scene Classifier") as demo:
        gr.HTML(
            """
            <header class="hero">
                <p class="hero-kicker">Geospatial intelligence · Scene classification</p>
                <h1>SatScene</h1>
                <p>Turn satellite imagery into a clear scene prediction, confidence-ranked
                alternatives, and a practical geospatial follow-up.</p>
            </header>
            """
        )
        with gr.Row(equal_height=True):
            with gr.Column(scale=5, elem_classes="panel"):
                gr.Markdown("### 01 · Add imagery", elem_classes="section-label")
                image = gr.Image(
                    type="filepath",
                    sources=["upload"],
                    label="Satellite or aerial image",
                    height=380,
                )
                gr.Markdown(
                    "Upload a clear overhead image. Your original image is not modified.",
                    elem_classes="helper-copy",
                )
            with gr.Column(scale=6, elem_classes="panel"):
                gr.Markdown("### 02 · Scene report", elem_classes="section-label")
                status = gr.Markdown(
                    "### Ready to analyze\nUpload an image, then select **Analyze image**.",
                    elem_classes="status-card",
                )
                predictions = gr.Label(
                    label="Confidence-ranked predictions",
                    num_top_classes=3,
                )
                follow_up = gr.Markdown(elem_classes="result-card")
        with gr.Row():
            run = gr.Button("Analyze image", variant="primary")
            clear = gr.Button("Clear all", variant="secondary")
        run.click(
            fn=lambda image_path: _analyze_image(agent, image_path),
            inputs=image,
            outputs=[status, predictions, follow_up],
        )
        clear.click(
            fn=_clear_results,
            inputs=None,
            outputs=[image, status, predictions, follow_up],
            queue=False,
        )
        gr.HTML(
            """
            """
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local SatScene GUI")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--share", action="store_true", help="create a public Gradio link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    checkpoint = args.checkpoint or os.path.join(cfg["output"]["dir"], "best_model.pth")
    if not Path(checkpoint).is_file():
        raise SystemExit(
            f"Checkpoint not found: {checkpoint}\n"
            "Download and prepare the dataset, then run `python -m satscene.train`, "
            "or pass a trained checkpoint with --checkpoint. See docs/MAC_SETUP.md."
        )

    device = best_device()
    metadata = checkpoint_metadata(checkpoint)
    names = _class_names(cfg, checkpoint)
    model_name = metadata.get("model_name", cfg["model"]["name"])
    img_size = int(metadata.get("img_size", cfg["data"]["img_size"]))
    model = build_model(len(names), model_name, pretrained=False, img_size=img_size)
    model = load_checkpoint(model, checkpoint, device)
    print(f"Loaded {len(names)} classes on {device}; GUI at http://127.0.0.1:{args.port}")

    demo = create_interface(SatelliteAgent(model, names, device, img_size))
    demo.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=args.share,
        inbrowser=True,
        show_error=True,
        theme=APP_THEME,
        css=APP_CSS,
    )


if __name__ == "__main__":
    main()
