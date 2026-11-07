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
from .model import build_model, checkpoint_metadata, checkpoint_not_found_message, load_checkpoint
from .runtime import best_device
from .ui_style import UI_COLORS, UI_COPY


APP_CSS = f"""
.gradio-container {{
    max-width: 1240px !important;
    margin: 0 auto !important;
    background: {UI_COLORS['page']} !important;
}}
.hero {{
    padding: 30px 34px;
    margin: 14px 0 22px;
    border-radius: 22px;
    color: {UI_COLORS['hero_text']};
    background: linear-gradient(
        115deg,
        {UI_COLORS['hero']} 0%,
        {UI_COLORS['hero_mid']} 55%,
        {UI_COLORS['hero_end']} 100%
    );
    box-shadow: 0 16px 36px rgba(20, 48, 75, .16);
}}
.hero-kicker {{
    margin: 0 0 8px;
    color: {UI_COLORS['kicker']};
    font-size: 12px;
    font-weight: 700;
    letter-spacing: .14em;
    text-transform: uppercase;
}}
.hero h1 {{
    margin: 0;
    color: {UI_COLORS['hero_text']};
    font-size: clamp(30px, 5vw, 46px);
    line-height: 1.08;
}}
.hero p {{
    max-width: 700px;
    margin: 12px 0 0;
    color: {UI_COLORS['hero_muted']};
    font-size: 16px;
    line-height: 1.6;
}}
.panel {{
    padding: 22px !important;
    border: 1px solid {UI_COLORS['border']} !important;
    border-radius: 18px !important;
    background: {UI_COLORS['surface']} !important;
    box-shadow: 0 8px 24px rgba(19, 47, 75, .055);
}}
.section-label {{
    color: {UI_COLORS['section']};
    font-size: 12px;
    font-weight: 700;
    letter-spacing: .1em;
    text-transform: uppercase;
}}
.helper-copy {{
    color: {UI_COLORS['subtle']};
    font-size: 13px;
    line-height: 1.55;
}}
.status-card {{
    min-height: 72px;
    padding: 14px 18px !important;
    border: 1px solid {UI_COLORS['status_border']} !important;
    border-radius: 14px !important;
    background: {UI_COLORS['status_bg']} !important;
}}
.status-card h3 {{
    margin: 0 0 5px;
    color: {UI_COLORS['text']};
}}
.status-card p {{
    margin: 0;
    color: {UI_COLORS['muted']};
}}
.result-card {{
    padding: 18px !important;
    border: 1px solid {UI_COLORS['result_border']} !important;
    border-radius: 14px !important;
    background: {UI_COLORS['result_bg']} !important;
}}
.disclaimer {{
    padding: 14px 18px;
    border-left: 3px solid {UI_COLORS['kicker']};
    border-radius: 8px;
    background: {UI_COLORS['status_bg']};
    color: {UI_COLORS['muted']};
    font-size: 13px;
    line-height: 1.55;
}}
@media (max-width: 700px) {{
    .hero {{ padding: 24px 20px; border-radius: 16px; }}
    .panel {{ padding: 16px !important; }}
}}
"""

APP_THEME = gr.themes.Soft().set(
    background_fill_primary_dark=UI_COLORS["page"],
    background_fill_secondary_dark=UI_COLORS["status_bg"],
    body_background_fill_dark=UI_COLORS["page"],
    body_text_color_dark=UI_COLORS["text"],
    body_text_color_subdued_dark=UI_COLORS["muted"],
    block_background_fill_dark=UI_COLORS["surface"],
    block_label_text_color_dark=UI_COLORS["section"],
    block_title_text_color_dark=UI_COLORS["text"],
    input_background_fill_dark=UI_COLORS["surface"],
    input_placeholder_color_dark=UI_COLORS["subtle"],
    button_secondary_background_fill_dark="#e8eef4",
    button_secondary_background_fill_hover_dark="#dce7ef",
    button_secondary_border_color_dark="#d2dee8",
    button_secondary_text_color_dark=UI_COLORS["text"],
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
        return "### Ready to analyze\nChoose an image before starting analysis.", {}, ""
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
        f"### {UI_COPY['follow_up']}\n"
        f"**{result['title']}**\n\n"
        f"{result['suggested_follow_up']}\n\n"
        f"{UI_COPY['follow_up_disclaimer']}"
    )
    return status, predictions, follow_up


def _analyze_models(
    agents: list[tuple[str, SatelliteAgent]],
    image_path: str | None,
) -> tuple[str | dict[str, float], ...]:
    if not image_path:
        return (
            f"### {UI_COPY['ready']}\n{UI_COPY['ready_detail']}",
            *({} for _ in agents),
        )

    predictions = []
    errors = []
    for name, agent in agents:
        try:
            result = agent.analyze_structured(image_path)
        except (OSError, ValueError, RuntimeError) as exc:
            predictions.append({})
            errors.append(f"{name}: {exc}")
        else:
            predictions.append({label: float(score) for label, score in result["top_k"]})

    if errors:
        status = "### Comparison completed with errors\n" + "\n".join(
            f"- {error}" for error in errors
        )
    else:
        status = f"### Comparison complete\nAll {len(agents)} models analyzed the image."
    return (status, *predictions)


def _image_ready(image_path: str | None, model_count: int = 1, comparison: bool = False):
    prediction_outputs = [None] * model_count
    if image_path:
        outputs = [
            "### Image ready\nSelect Analyze image to view ranked predictions.",
            *prediction_outputs,
        ]
    else:
        outputs = [
            f"### {UI_COPY['ready']}\n{UI_COPY['ready_detail']}",
            *prediction_outputs,
        ]
    if not comparison:
        outputs.append(UI_COPY["follow_up_placeholder"])
    outputs.append(gr.update(interactive=bool(image_path)))
    return tuple(outputs)


def _clear_results(model_count: int = 1, comparison: bool = False):
    outputs = [
        None,
        f"### {UI_COPY['ready']}\n{UI_COPY['ready_detail']}",
        *([None] * model_count),
    ]
    if not comparison:
        outputs.append("")
    outputs.append(gr.update(interactive=False))
    return tuple(outputs)


def create_interface(
    agent: SatelliteAgent,
    comparison_agents: list[tuple[str, SatelliteAgent]] | None = None,
):
    """Build the upload-and-predict interface without starting a web server."""
    is_comparison = comparison_agents is not None
    result_count = len(comparison_agents) if is_comparison else 1
    with gr.Blocks(title="SatScene · Satellite Scene Classifier") as demo:
        gr.HTML(
            f"""
            <header class="hero">
                <p class="hero-kicker">{UI_COPY['kicker']}</p>
                <h1>{UI_COPY['title']}</h1>
                <p>{UI_COPY['intro']}</p>
            </header>
            """
        )
        with gr.Row(equal_height=True):
            with gr.Column(scale=5, elem_classes="panel"):
                gr.Markdown(f"### {UI_COPY['add_image']}", elem_classes="section-label")
                image = gr.Image(
                    type="filepath",
                    sources=["upload"],
                    label=UI_COPY["image_label"],
                    height=380,
                )
                gr.Markdown(
                    UI_COPY["image_helper"],
                    elem_classes="helper-copy",
                )
                with gr.Row():
                    run = gr.Button(UI_COPY["analyze"], variant="primary", interactive=False)
                    clear = gr.Button(UI_COPY["clear"], variant="secondary")
            with gr.Column(scale=6, elem_classes="panel"):
                gr.Markdown(f"### {UI_COPY['scene_report']}", elem_classes="section-label")
                status = gr.Markdown(
                    f"### {UI_COPY['ready']}\n{UI_COPY['ready_detail']}",
                    elem_classes="status-card",
                )
                if is_comparison:
                    prediction_outputs = []
                    with gr.Row():
                        for name, _ in comparison_agents:
                            with gr.Column():
                                gr.Markdown(f"#### {name}")
                                prediction_outputs.append(
                                    gr.Label(
                                        label=UI_COPY["predictions"],
                                        num_top_classes=3,
                                    )
                                )
                else:
                    prediction_outputs = [
                        gr.Label(
                            label=UI_COPY["predictions"],
                            num_top_classes=3,
                        )
                    ]
                follow_up = None
                if not is_comparison:
                    follow_up = gr.Markdown(
                        UI_COPY["follow_up_placeholder"],
                        elem_classes="result-card",
                    )
            image.change(
                fn=lambda image_path: _image_ready(
                    image_path,
                    model_count=result_count,
                    comparison=is_comparison,
                ),
                inputs=image,
                outputs=[status, *prediction_outputs]
                + ([] if is_comparison else [follow_up])
                + [run],
                queue=False,
            )
        analyze = (
            lambda image_path: _analyze_models(comparison_agents, image_path)
            if is_comparison
            else _analyze_image(agent, image_path)
        )
        run.click(
            fn=analyze,
            inputs=image,
            outputs=[status, *prediction_outputs]
            + ([] if is_comparison else [follow_up]),
        )
        clear.click(
            fn=lambda: _clear_results(
                model_count=result_count,
                comparison=is_comparison,
            ),
            inputs=None,
            outputs=[image, status, *prediction_outputs]
            + ([] if is_comparison else [follow_up])
            + [run],
            queue=False,
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local SatScene GUI")
    parser.add_argument("--config", default="configs/default.yaml")
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--checkpoint", default=None)
    checkpoint_group.add_argument(
        "--compare-checkpoints",
        nargs=3,
        metavar=("MODEL_1", "MODEL_2", "MODEL_3"),
        help="compare exactly three model checkpoints on each uploaded image",
    )
    parser.add_argument("--share", action="store_true", help="create a public Gradio link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    device = best_device()
    if args.compare_checkpoints:
        comparison_agents = []
        for checkpoint in args.compare_checkpoints:
            if not Path(checkpoint).is_file():
                raise SystemExit(
                    checkpoint_not_found_message(checkpoint)
                    + "\nSee docs/MAC_SETUP.md for model-specific commands."
                )
            metadata = checkpoint_metadata(checkpoint)
            names = _class_names(cfg, checkpoint)
            model_name = metadata.get("model_name", cfg["model"]["name"])
            img_size = int(metadata.get("img_size", cfg["data"]["img_size"]))
            model = build_model(len(names), model_name, pretrained=False, img_size=img_size)
            model = load_checkpoint(model, checkpoint, device)
            comparison_agents.append(
                (model_name, SatelliteAgent(model, names, device, img_size))
            )
            print(f"Loaded {model_name} ({len(names)} classes) on {device}")
        demo = create_interface(comparison_agents[0][1], comparison_agents)
    else:
        checkpoint = args.checkpoint or os.path.join(cfg["output"]["dir"], "best_model.pth")
        if not Path(checkpoint).is_file():
            raise SystemExit(
                checkpoint_not_found_message(checkpoint)
                + "\nSee docs/MAC_SETUP.md for model-specific commands."
            )
        metadata = checkpoint_metadata(checkpoint)
        names = _class_names(cfg, checkpoint)
        model_name = metadata.get("model_name", cfg["model"]["name"])
        img_size = int(metadata.get("img_size", cfg["data"]["img_size"]))
        model = build_model(len(names), model_name, pretrained=False, img_size=img_size)
        model = load_checkpoint(model, checkpoint, device)
        print(f"Loaded {len(names)} classes on {device}")
        demo = create_interface(SatelliteAgent(model, names, device, img_size))

    print(f"GUI at http://127.0.0.1:{args.port}")
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
