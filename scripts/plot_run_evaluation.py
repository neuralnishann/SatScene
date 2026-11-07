"""Build a vector dashboard from the archived comparison run metrics."""
from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "ConvNeXtV2-Tiny": ("convnextv2_tiny", "#0072B2"),
    "SwinV2-Tiny": ("swinv2_tiny_window8", "#D55E00"),
    "DINOv2 ViT-S/14": ("dinov2_vit_small_patch14", "#009E73"),
}
OUTPUT = ROOT / "results" / "figures" / "run_evaluation_dashboard.svg"
WIDTH, HEIGHT = 1740, 1120


def text(x: float, y: float, value: str, size: int = 18, **attrs: str) -> str:
    size = round(size * (1.5 if size <= 18 else 1.05))
    extra = " ".join(f'{key.replace("_", "-")}="{html.escape(val)}"' for key, val in attrs.items())
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" {extra}>'
        f"{html.escape(value)}</text>"
    )


def load_run(folder: str, split: str) -> dict:
    path = ROOT / "runs" / folder / f"metrics_{split}.json"
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def paired_validation_counts() -> tuple[dict[tuple[bool, bool], int], int]:
    predictions: dict[str, dict[str, tuple[bool, str]]] = {}
    for label, (folder, _) in RUNS.items():
        path = ROOT / "runs" / folder / "predictions_valid.csv"
        with path.open(newline="", encoding="utf-8") as file:
            rows = csv.DictReader(file)
            predictions[label] = {
                Path(row.get("image_path") or row["image"]).name: (
                    row["true_class"] == row["predicted_class"],
                    row["true_class"],
                )
                for row in rows
            }

    conv = predictions["ConvNeXtV2-Tiny"]
    swin = predictions["SwinV2-Tiny"]
    if conv.keys() != swin.keys():
        raise ValueError("ConvNeXt and Swin validation predictions do not match by image")
    if any(conv[key][1] != swin[key][1] for key in conv):
        raise ValueError("Validation ground-truth labels differ between model runs")

    counts = {(conv_ok, swin_ok): 0 for conv_ok in (False, True) for swin_ok in (False, True)}
    for key in conv:
        counts[(conv[key][0], swin[key][0])] += 1
    expected_n = load_run("convnextv2_tiny", "valid")["n"]
    if len(conv) != expected_n:
        raise ValueError(f"Expected {expected_n} matched validation images; found {len(conv)}")
    return counts, len(conv)


def build_svg() -> str:
    metrics = {
        label: {
            "test": load_run(folder, "test"),
        }
        for label, (folder, _) in RUNS.items()
    }
    class_names = list(metrics["ConvNeXtV2-Tiny"]["test"]["per_class"])
    supports = {
        name: int(metrics["ConvNeXtV2-Tiny"]["test"]["per_class"][name]["support"])
        for name in class_names
    }
    for run in metrics.values():
        if list(run["test"]["per_class"]) != class_names:
            raise ValueError("Class order differs between test reports")
        if any(int(run["test"]["per_class"][name]["support"]) != supports[name] for name in class_names):
            raise ValueError("Test class supports differ between model runs")
        report = run["test"]
        if sum(supports.values()) != int(report["n"]):
            raise ValueError("Test class supports do not sum to the reported sample count")
        bins = report["calibration_bins"]
        if sum(int(item["count"]) for item in bins) != int(report["n"]):
            raise ValueError("Calibration-bin counts do not cover the test split")
        ece_from_bins = sum(
            int(item["count"]) / int(report["n"])
            * abs(float(item["accuracy"]) - float(item["confidence"]))
            for item in bins
        )
        if not math.isclose(ece_from_bins, float(report["ece_top_label"]), abs_tol=1e-10):
            raise ValueError("Saved ECE does not match the saved reliability bins")

    paired, paired_n = paired_validation_counts()
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        """<style>
          text{font-family:Arial,sans-serif;fill:#18324b}
          .muted{fill:#52677b}.grid{stroke:#dbe3eb;stroke-width:1}
          .panel{fill:#fff;stroke:#c8d3df;stroke-width:1.5}
          .axis{stroke:#49617a;stroke-width:1.4}
        </style>""",
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#f7f9fc"/>',
        text(38, 48, "Run-level evaluation: class performance, calibration, and paired errors", 29, font_weight="700"),
        text(38, 78, "All test metrics use the same 100-image split; the paired correctness matrix uses the 200-image validation split.", 17, class_="muted"),
        '<rect x="28" y="95" width="990" height="970" rx="14" class="panel"/>',
        '<rect x="1032" y="95" width="680" height="510" rx="14" class="panel"/>',
        '<rect x="1032" y="620" width="680" height="445" rx="14" class="panel"/>',
        text(52, 136, "A. Test per-class F1 (point = score; right column = support)", 21, font_weight="700"),
    ]

    plot_left, plot_right = 355, 900
    plot_top, row_height = 205, 48
    for tick in range(6):
        value = tick / 5
        x = plot_left + value * (plot_right - plot_left)
        elements.append(f'<line x1="{x:.1f}" y1="170" x2="{x:.1f}" y2="890" class="grid"/>')
        elements.append(text(x, 164, f"{value:.1f}", 14, text_anchor="middle", class_="muted"))
    elements.append(f'<line x1="{plot_left}" y1="174" x2="{plot_left}" y2="890" class="axis"/>')
    elements.append(f'<line x1="{plot_left}" y1="890" x2="{plot_right}" y2="890" class="axis"/>')
    elements.append(text(931, 164, "n", 15, text_anchor="middle", font_weight="700"))

    offsets = {"ConvNeXtV2-Tiny": -8, "SwinV2-Tiny": 0, "DINOv2 ViT-S/14": 8}
    for index, name in enumerate(class_names):
        y = plot_top + index * row_height
        elements.append(text(337, y + 6, name, 16, text_anchor="end"))
        elements.append(text(931, y + 6, str(supports[name]), 15, text_anchor="middle", class_="muted"))
        if index % 2 == 0:
            elements.append(f'<rect x="345" y="{y - 16}" width="570" height="32" fill="#eef3f8" opacity=".55"/>')
        for label, (_, color) in RUNS.items():
            score = float(metrics[label]["test"]["per_class"][name]["f1-score"])
            x = plot_left + score * (plot_right - plot_left)
            cy = y + offsets[label]
            elements.append(
                f'<circle cx="{x:.1f}" cy="{cy:.1f}" r="6.5" fill="{color}" '
                f'stroke="white" stroke-width="1.5"><title>{html.escape(label)}: '
                f'F1={score:.3f}; support={supports[name]}</title></circle>'
            )
    legend_y = 940
    for i, (label, (_, color)) in enumerate(RUNS.items()):
        x = 95 + i * 250
        elements.append(f'<circle cx="{x}" cy="{legend_y}" r="7" fill="{color}"/>')
        elements.append(text(x + 14, legend_y + 6, label, 15))
    elements.append(text(52, 1000, "F1 is unstable for rare classes: Lake n=2 and River n=1.", 16, class_="muted"))
    elements.append(text(52, 1026, "Scores are descriptive point estimates, not class-level confidence intervals.", 16, class_="muted"))

    elements.extend([
        text(1056, 136, "B. Test top-label reliability", 21, font_weight="700"),
        text(1056, 161, "Marker area increases with the number of examples in each populated bin.", 10, class_="muted"),
    ])
    x0, x1, y0, y1 = 1100, 1632, 205, 480
    for tick in range(6):
        value = tick / 5
        x = x0 + value * (x1 - x0)
        y = y1 - value * (y1 - y0)
        elements.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y1}" class="grid"/>')
        elements.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" class="grid"/>')
        elements.append(text(x, y1 + 22, f"{value:.1f}", 13, text_anchor="middle", class_="muted"))
        elements.append(text(x0 - 12, y + 5, f"{value:.1f}", 13, text_anchor="end", class_="muted"))
    elements.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y0}" stroke="#64748b" stroke-dasharray="7 6" stroke-width="2"/>')
    elements.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" class="axis"/>')
    elements.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" class="axis"/>')
    for label, (_, color) in RUNS.items():
        bins = metrics[label]["test"]["calibration_bins"]
        points = [
            (x0 + float(item["confidence"]) * (x1 - x0),
             y1 - float(item["accuracy"]) * (y1 - y0))
            for item in bins
        ]
        if len(points) > 1:
            points_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            elements.append(f'<polyline points="{points_str}" fill="none" stroke="{color}" stroke-width="2"/>')
        for item, (x, y) in zip(bins, points):
            radius = min(11, 3.5 + math.sqrt(int(item["count"])))
            elements.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{color}" '
                f'fill-opacity=".78" stroke="white" stroke-width="1.3"><title>{html.escape(label)}: '
                f'confidence={item["confidence"]:.3f}; accuracy={item["accuracy"]:.3f}; '
                f'n={item["count"]}</title></circle>'
            )
    for i, (label, (_, color)) in enumerate(RUNS.items()):
        score = float(metrics[label]["test"]["ece_top_label"])
        y = 540 + i * 24
        elements.append(f'<circle cx="1070" cy="{y - 5}" r="6" fill="{color}"/>')
        elements.append(text(1082, y, f"{label} ECE={score:.3f}", 14))
    elements.append(text((x0 + x1) / 2, 190, "Mean confidence", 11, text_anchor="middle", class_="muted"))

    elements.extend([
        text(1056, 661, "C. Paired validation correctness: ConvNeXt vs Swin", 21, font_weight="700"),
        text(1056, 689, "Each cell counts the same image evaluated by both models (N=200).", 10, class_="muted"),
    ])
    cell_w, cell_h, mx, my = 155, 80, 1350, 760
    elements.append(text(mx + cell_w / 2, my - 27, "Swin correct", 15, text_anchor="middle"))
    elements.append(text(mx + cell_w * 1.5 + 8, my - 27, "Swin wrong", 15, text_anchor="middle"))
    elements.append(text(mx - 155, my + cell_h / 2 + 5, "ConvNeXt correct", 14, text_anchor="middle"))
    elements.append(text(mx - 155, my + cell_h * 1.5 + 5, "ConvNeXt wrong", 14, text_anchor="middle"))
    cell_positions = [
        (True, True, mx, my),
        (True, False, mx + cell_w + 12, my),
        (False, True, mx, my + cell_h + 12),
        (False, False, mx + cell_w + 12, my + cell_h + 12),
    ]
    for conv_ok, swin_ok, x, y in cell_positions:
        count = paired[(conv_ok, swin_ok)]
        fill = "#dcefe9" if conv_ok == swin_ok else "#fff0d9"
        elements.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" rx="7" fill="{fill}" stroke="#b8c8d8"/>')
        elements.append(text(x + cell_w / 2, y + 34, str(count), 26, text_anchor="middle", font_weight="700"))
        status = "both correct" if conv_ok else "both wrong"
        if conv_ok != swin_ok:
            status = "ConvNeXt only" if conv_ok else "Swin only"
        elements.append(text(x + cell_w / 2, y + 59, status, 13, text_anchor="middle", class_="muted"))
    both_right = paired[(True, True)]
    both_wrong = paired[(False, False)]
    conv_only = paired[(True, False)]
    swin_only = paired[(False, True)]
    diff = (conv_only - swin_only) / paired_n
    elements.append(text(1056, 980, f"Paired accuracy difference = (12 - 11)/200 = {diff:.3f}", 17, font_weight="700"))
    elements.append(text(1056, 1010, f"Both correct: {both_right}; both wrong: {both_wrong}; discordant: {conv_only + swin_only}.", 15, class_="muted"))
    elements.append(text(1056, 1037, "Validation was used for checkpoint selection; paired counts are descriptive.", 10, class_="muted"))
    elements.append("</svg>")
    return "\n".join(elements)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(build_svg(), encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
