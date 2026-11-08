"""Build the editable SatScene presentation in papers/presentation.pptx."""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "papers" / "presentation.pptx"
DEMO_IMAGE = ROOT / "papers" / "demo_ui.png"
SLIDE_W, SLIDE_H = 13.333, 7.5

NAVY = "12263A"
INK = "183248"
TEAL = "0D7280"
MINT = "DFF3EF"
BLUE = "3973AC"
PALE = "F3F7FA"
WHITE = "FFFFFF"
MUTED = "627587"
LINE = "D6E1E9"
GOLD = "E6A23C"


def rgb(value: str) -> RGBColor:
    return RGBColor.from_string(value)


def add_box(
    slide,
    x: float,
    y: float,
    w: float,
    h: float,
    fill: str,
    *,
    rounded: bool = True,
    outline: str | None = None,
):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(fill)
    if outline:
        shape.line.color.rgb = rgb(outline)
        shape.line.width = Pt(1)
    else:
        shape.line.fill.background()
    return shape


def add_text(
    slide,
    text: str,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    size: float = 16,
    color: str = INK,
    bold: bool = False,
    align=PP_ALIGN.LEFT,
    valign=MSO_ANCHOR.TOP,
    font: str = "Aptos",
    margin: float = 0.02,
):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = Inches(margin)
    frame.margin_right = Inches(margin)
    frame.margin_top = Inches(margin)
    frame.margin_bottom = Inches(margin)
    frame.vertical_anchor = valign
    for index, line in enumerate(text.split("\n")):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.text = line
        paragraph.alignment = align
        paragraph.space_after = Pt(5)
        paragraph.font.name = font
        paragraph.font.size = Pt(size)
        paragraph.font.bold = bold
        paragraph.font.color.rgb = rgb(color)
    return shape


def add_bullets(slide, lines: list[str], x: float, y: float, w: float, h: float,
                *, size: float = 16, color: str = INK, gap: float = 0.42):
    for index, line in enumerate(lines):
        cy = y + index * gap
        add_box(slide, x, cy + 0.11, 0.09, 0.09, TEAL, rounded=False)
        add_text(slide, line, x + 0.22, cy, w - 0.22, gap,
                 size=size, color=color)


def add_header(slide, eyebrow: str, title: str, subtitle: str, page: int, source: str):
    add_text(slide, eyebrow.upper(), 0.68, 0.25, 10.8, 0.25,
             size=9, color=TEAL, bold=True)
    add_text(slide, title, 0.68, 0.57, 12.0, 0.55,
             size=27, color=NAVY, bold=True)
    add_text(slide, subtitle, 0.7, 1.18, 12.0, 0.38,
             size=12, color=MUTED)
    add_text(slide, source, 0.68, 7.12, 11.6, 0.18,
             size=8, color=MUTED)
    add_text(slide, f"{page:02d}", 12.15, 7.08, 0.5, 0.25,
             size=9, color=TEAL, bold=True, align=PP_ALIGN.RIGHT)


def new_content_slide(pres: Presentation, eyebrow: str, title: str, subtitle: str,
                      page: int, source: str):
    slide = pres.slides.add_slide(pres.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = rgb(WHITE)
    add_header(slide, eyebrow, title, subtitle, page, source)
    return slide


def add_cell_text(cell, text: str, *, size: int = 12, color: str = INK,
                  bold: bool = False, align=PP_ALIGN.CENTER):
    cell.text = str(text)
    cell.margin_left = Inches(0.06)
    cell.margin_right = Inches(0.06)
    cell.margin_top = Inches(0.04)
    cell.margin_bottom = Inches(0.04)
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    for paragraph in cell.text_frame.paragraphs:
        paragraph.alignment = align
        for run in paragraph.runs:
            run.font.name = "Aptos"
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.color.rgb = rgb(color)


def build_deck() -> None:
    if not DEMO_IMAGE.is_file():
        raise FileNotFoundError(
            f"Missing live application screenshot: {DEMO_IMAGE}. "
            "Capture the GUI demo before building the presentation."
        )

    pres = Presentation()
    pres.slide_width = Inches(SLIDE_W)
    pres.slide_height = Inches(SLIDE_H)
    pres.core_properties.title = "SatScene: Scene Classification from Overhead Imagery"
    pres.core_properties.subject = "Project overview, method, results, and software demonstration"
    pres.core_properties.author = "S M Gazzali Arafat Nishan"
    pres.core_properties.keywords = "SatScene, remote sensing, image classification, demo"

    # 1. Cover
    slide = pres.slides.add_slide(pres.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = rgb(NAVY)
    add_box(slide, 8.95, -0.75, 5.1, 4.2, TEAL)
    add_box(slide, 9.55, 4.25, 4.5, 3.8, BLUE)
    add_text(slide, "GEOSPATIAL DATA SCIENCE · SOFTWARE DEMO", 0.8, 0.8, 8, 0.35,
             size=11, color="A7DCD8", bold=True)
    add_text(slide, "SatScene", 0.78, 1.55, 8.6, 0.95,
             size=54, color=WHITE, bold=True)
    add_text(slide, "Seeing scenes from above", 0.83, 2.67, 8.1, 0.62,
             size=31, color=WHITE, bold=True)
    add_text(slide, "A documented 14-class comparison of pretrained vision models",
             0.85, 3.43, 7.7, 0.78, size=19, color="DCE9F1")
    add_box(slide, 0.84, 4.72, 3.1, 0.55, "214657")
    add_text(slide, "METHOD  ·  RESULTS  ·  LIVE GUI", 1.02, 4.86, 2.8, 0.22,
             size=10, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, "S M Gazzali Arafat Nishan\nFaculty of Computer Science · HSE University",
             0.85, 6.05, 7.8, 0.7, size=13, color="DCE9F1")
    add_text(slide, "Based on the project paper · November 2026", 0.85, 7.1, 8, 0.2,
             size=9, color="A7DCD8")

    # 2. Problem and goal
    slide = new_content_slide(
        pres, "01 · Problem", "From image piles to a useful first-pass label",
        "Scene classification can help triage overhead imagery—but it is not a map or a GIS measurement.",
        2, "Source: paper/main.tex, Introduction and Limitations and Future Work.",
    )
    add_box(slide, 0.75, 1.85, 5.75, 2.6, PALE, outline=LINE)
    add_text(slide, "THE PROBLEM", 1.05, 2.12, 4.8, 0.3,
             size=11, color=TEAL, bold=True)
    add_text(slide, "Overhead image collections are hard to scan one image at a time.",
             1.05, 2.55, 4.95, 0.95, size=22, color=NAVY, bold=True)
    add_text(slide, "Labels are often object annotations—not verified scene labels—and inputs may have no spatial metadata.",
             1.05, 3.57, 4.95, 0.62, size=14, color=MUTED)
    add_box(slide, 6.82, 1.85, 5.75, 2.6, MINT)
    add_text(slide, "THE PROJECT", 7.12, 2.12, 4.8, 0.3,
             size=11, color=TEAL, bold=True)
    add_text(slide, "Build an inspectable image-to-report workflow.",
             7.12, 2.55, 4.95, 0.95, size=22, color=NAVY, bold=True)
    add_text(slide, "Compare three pretrained classifiers and provide ranked predictions plus a static analyst follow-up.",
             7.12, 3.57, 4.95, 0.62, size=14, color=MUTED)
    add_text(slide, "INTENDED OUTPUT", 0.85, 4.9, 3.5, 0.3,
             size=10, color=TEAL, bold=True)
    add_bullets(slide, [
        "A predicted scene class and confidence-ranked alternatives",
        "A lightweight aid for analyst review—not an automated decision",
        "A reproducible, local demonstration of model inference",
    ], 0.9, 5.35, 11.7, 1.35, size=15, gap=0.46)

    # 3. Input, label construction, output
    slide = new_content_slide(
        pres, "02 · Data", "What goes in—and what comes out?",
        "The target is derived from YOLO object labels; it is a proxy for the image-level scene.",
        3, "Source: paper/main.tex, Dataset and Target Construction; Kaggle / Roboflow export v2.",
    )
    cards = [
        (0.72, "INPUT", "1,000 RGB JPEGs\n700 train · 200 validation\n100 test",
         "14 categories\nNo coordinates, CRS, geotransform, or multispectral bands."),
        (4.72, "TARGET RULE", "Count YOLO object classes\nChoose the most frequent",
         "Ties go to the first class in the annotation file. Missing or empty labels are skipped."),
        (8.72, "APP OUTPUT", "Top scene class\nConfidence-ranked alternatives",
         "A static class-associated follow-up suggestion. It does not retrieve GIS data or run GIS analysis."),
    ]
    for x, label, lead, detail in cards:
        add_box(slide, x, 1.9, 3.8, 2.65, PALE, outline=LINE)
        add_text(slide, label, x + 0.25, 2.15, 3.2, 0.3,
                 size=10, color=TEAL, bold=True)
        add_text(slide, lead, x + 0.25, 2.62, 3.22, 0.85,
                 size=18, color=NAVY, bold=True)
        add_text(slide, detail, x + 0.25, 3.6, 3.22, 0.72,
                 size=12, color=MUTED)
    add_text(slide, "14 CLASSES", 0.83, 4.9, 1.5, 0.25,
             size=10, color=TEAL, bold=True)
    add_text(slide,
             "Agriculture · Airport · Beach · City · Desert · Forest · Grassland · Highway · "
             "Lake · Mountain · Parking · Port · Railway · River",
             0.83, 5.28, 11.7, 0.72, size=14, color=INK)
    add_box(slide, 0.82, 6.17, 11.7, 0.55, "FFF4E1")
    add_text(slide, "Interpretation guardrail: dominant detected object ≠ verified land-use scene.",
             1.05, 6.34, 11.1, 0.22, size=12, color=NAVY, bold=True)

    # 4. Method
    slide = new_content_slide(
        pres, "03 · Method", "A validation-selected, three-backbone comparison",
        "The paper’s comparison notebook and packaged single-model trainer are separate workflows.",
        4, "Source: paper/main.tex, Methods and Experimental Setup.",
    )
    steps = [
        ("01", "Parse labels", "YOLO boxes → one image target"),
        ("02", "Prepare image", "RGB · resize · ImageNet normalization"),
        ("03", "Fine-tune", "Seed 42 · 10 epochs · AdamW"),
        ("04", "Select", "Best validation macro-F1 checkpoint"),
        ("05", "Evaluate", "Held-out split · report metrics"),
    ]
    x_positions = [0.72, 3.22, 5.72, 8.22, 10.72]
    for x, (number, title, detail) in zip(x_positions, steps):
        add_box(slide, x, 2.0, 2.1, 1.75, PALE, outline=LINE)
        add_text(slide, number, x + 0.16, 2.16, 0.55, 0.32,
                 size=12, color=TEAL, bold=True)
        add_text(slide, title, x + 0.16, 2.58, 1.78, 0.36,
                 size=16, color=NAVY, bold=True)
        add_text(slide, detail, x + 0.16, 3.03, 1.78, 0.55,
                 size=10, color=MUTED)
        if x != x_positions[-1]:
            add_text(slide, "›", x + 2.16, 2.6, 0.3, 0.4,
                     size=25, color=TEAL, bold=True)
    add_text(slide, "PRETRAINED BACKBONES", 0.85, 4.18, 3.5, 0.28,
             size=10, color=TEAL, bold=True)
    model_cards = [
        ("ConvNeXtV2-Tiny", "224 px", "Selected epoch 9"),
        ("SwinV2-Tiny", "256 px", "Selected epoch 10"),
        ("DINOv2 ViT-S/14", "224 px", "Selected epoch 10"),
    ]
    for x, (name, size, selected) in zip([0.85, 4.85, 8.85], model_cards):
        add_box(slide, x, 4.62, 3.55, 1.0, MINT)
        add_text(slide, name, x + 0.2, 4.8, 3.1, 0.3,
                 size=16, color=NAVY, bold=True)
        add_text(slide, f"{size} input · {selected}", x + 0.2, 5.2, 3.1, 0.22,
                 size=11, color=MUTED)
    add_text(slide, "Batch 16 · train-only flips and 90° rotations (p = 0.5 each) · LR 3×10⁻⁴ · weight decay 10⁻⁴",
             0.86, 6.08, 11.5, 0.38, size=12, color=INK)

    # 5. Results
    slide = new_content_slide(
        pres, "04 · Results", "ConvNeXtV2-Tiny leads this one-run comparison",
        "Validation selected checkpoints; the same supplied 100-image test split has since been evaluated.",
        5, "Source: paper/main.tex, Results; single run per architecture (seed 42).",
    )
    chart_data = CategoryChartData()
    chart_data.categories = ["ConvNeXtV2", "SwinV2", "DINOv2"]
    chart_data.add_series("Validation", (0.800, 0.795, 0.380))
    chart_data.add_series("Test", (0.860, 0.800, 0.410))
    chart = slide.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED,
        Inches(0.62), Inches(1.82), Inches(6.2), Inches(4.05), chart_data,
    ).chart
    chart.has_title = True
    chart.chart_title.text_frame.text = "Top-1 accuracy"
    for paragraph in chart.chart_title.text_frame.paragraphs:
        for run in paragraph.runs:
            run.font.name = "Aptos"
            run.font.size = Pt(14)
            run.font.bold = True
            run.font.color.rgb = rgb(NAVY)
    chart.has_legend = True
    chart.legend.position = XL_LEGEND_POSITION.BOTTOM
    chart.legend.include_in_layout = False
    chart.value_axis.minimum_scale = 0
    chart.value_axis.maximum_scale = 1
    chart.value_axis.major_unit = 0.2
    chart.value_axis.tick_labels.number_format = "0%"
    chart.value_axis.has_major_gridlines = True
    chart.category_axis.tick_labels.font.name = "Aptos"
    chart.category_axis.tick_labels.font.size = Pt(10)
    chart.value_axis.tick_labels.font.name = "Aptos"
    chart.value_axis.tick_labels.font.size = Pt(9)
    for series, color in zip(chart.series, [TEAL, GOLD]):
        series.format.fill.solid()
        series.format.fill.fore_color.rgb = rgb(color)
        series.format.line.fill.background()
        series.has_data_labels = True
        series.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
        series.data_labels.number_format = "0%"
        series.data_labels.font.name = "Aptos"
        series.data_labels.font.size = Pt(9)
        series.data_labels.font.color.rgb = rgb(INK)

    add_text(slide, "MACRO-F1", 7.05, 1.92, 3.2, 0.3,
             size=11, color=TEAL, bold=True)
    table_shape = slide.shapes.add_table(
        4, 3, Inches(7.02), Inches(2.35), Inches(5.55), Inches(2.25),
    )
    table = table_shape.table
    table.columns[0].width = Inches(2.55)
    table.columns[1].width = Inches(1.5)
    table.columns[2].width = Inches(1.5)
    rows = [
        ("Model", "Validation", "Test"),
        ("ConvNeXtV2-Tiny", "0.788", "0.783"),
        ("SwinV2-Tiny", "0.782", "0.758"),
        ("DINOv2 ViT-S/14", "0.305", "0.340"),
    ]
    for r, values in enumerate(rows):
        for c, value in enumerate(values):
            cell = table.cell(r, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(NAVY if r == 0 else (PALE if r % 2 else WHITE))
            add_cell_text(cell, value, size=11 if r else 10,
                          color=WHITE if r == 0 else INK,
                          bold=(r == 0 or (r == 1 and c == 0)),
                          align=PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER)
    add_box(slide, 7.02, 4.88, 5.55, 1.32, PALE, outline=LINE)
    add_text(slide, "WHAT TO TAKE AWAY", 7.28, 5.06, 4.9, 0.25,
             size=10, color=TEAL, bold=True)
    add_text(slide, "ConvNeXtV2 led on these splits. The validation gap over SwinV2 was one image.",
             7.28, 5.39, 4.95, 0.65, size=14, color=NAVY, bold=True)
    add_text(slide, "Test ConvNeXtV2: 86/100 (95% Wilson CI 77.9–91.5%). Descriptive only: one run per model; 1-image validation gap.",
             0.85, 6.3, 11.7, 0.3, size=11, color=MUTED, bold=True)

    # 6. Live software demo
    slide = new_content_slide(
        pres, "05 · Software demo", "One image in → ranked prediction + analyst prompt",
        "Live local browser run using the project’s included Desert sample image.",
        6, "Demo: local Gradio app + iodata/sample_input/015_…jpg; run 8 Nov 2026.",
    )
    slide.shapes.add_picture(str(DEMO_IMAGE), Inches(0.65), Inches(1.78),
                             height=Inches(5.05))
    add_box(slide, 9.05, 1.86, 3.58, 4.82, PALE, outline=LINE)
    add_text(slide, "DEMO WALKTHROUGH", 9.32, 2.14, 3.0, 0.28,
             size=10, color=TEAL, bold=True)
    add_bullets(slide, [
        "Upload the overhead image",
        "Select Analyze image",
        "Read the top class and alternatives",
        "Review the suggested follow-up",
    ], 9.34, 2.58, 3.0, 1.95, size=13, gap=0.48)
    add_box(slide, 9.3, 4.78, 3.02, 0.82, MINT)
    add_text(slide, "Observed: Desert · UI rounds confidence to 100%",
             9.48, 4.96, 2.68, 0.45, size=12, color=NAVY, bold=True)
    add_text(slide, "Follow-up is static advice—not a GIS analysis. This is one example, not evidence of reliability.",
             9.34, 5.85, 2.98, 0.62, size=11, color=MUTED)

    # 7. Reproduce / launch
    slide = new_content_slide(
        pres, "06 · Run it yourself", "Launch the local GUI with a checkpoint",
        "Run from the repository root with the project environment active.",
        7, "Source: docs/MAC_SETUP.md and src/satscene/app.py.",
    )
    add_box(slide, 0.78, 1.92, 7.45, 2.05, NAVY)
    add_text(slide, "BROWSER GUI", 1.05, 2.16, 2.0, 0.25,
             size=10, color="A7DCD8", bold=True)
    add_text(slide, "python -m satscene.app --config configs/default.yaml \\\n"
             "  --checkpoint runs/convnextv2_tiny/best_model.pth",
             1.05, 2.56, 6.8, 0.95, size=15, color=WHITE, font="Menlo")
    add_box(slide, 0.78, 4.2, 7.45, 1.45, PALE, outline=LINE)
    add_text(slide, "NATIVE MACOS GUI", 1.05, 4.43, 2.5, 0.25,
             size=10, color=TEAL, bold=True)
    add_text(slide, "python -m satscene.desktop --config configs/default.yaml \\\n"
             "  --checkpoint runs/convnextv2_tiny/best_model.pth",
             1.05, 4.81, 6.8, 0.65, size=13, color=NAVY, font="Menlo")
    add_box(slide, 8.62, 1.92, 3.95, 3.73, MINT)
    add_text(slide, "CHECKPOINT NOTE", 8.93, 2.22, 3.2, 0.28,
             size=10, color=TEAL, bold=True)
    add_text(slide,
             "The example uses a local comparison checkpoint. These large .pth files "
             "are Git-ignored; a fresh clone may not have them.",
             8.93, 2.68, 3.1, 1.1, size=15, color=NAVY, bold=True)
    add_text(slide,
             "Train the packaged model first, or pass a checkpoint you already have. "
             "The app’s default path is runs/convnext_tiny/best_model.pth.",
             8.93, 3.95, 3.1, 1.05, size=12, color=MUTED)
    add_box(slide, 8.62, 5.95, 3.95, 0.72, "FFF4E1")
    add_text(slide, "Browser: 127.0.0.1:7860 · Desktop: macOS only",
             8.82, 6.17, 3.55, 0.25, size=10, color=NAVY, bold=True)

    # 8. Limitations and next steps
    slide = new_content_slide(
        pres, "07 · Interpretation", "Promising workflow; bounded evidence",
        "The paper’s main contribution is an inspectable baseline—not a claim of operational generalization.",
        8, "Source: paper/main.tex, Limitations and Future Work; Conclusion.",
    )
    add_box(slide, 0.78, 1.9, 5.8, 3.85, PALE, outline=LINE)
    add_text(slide, "LIMITS", 1.08, 2.2, 4.8, 0.3,
             size=11, color=TEAL, bold=True)
    add_bullets(slide, [
        "Proxy labels from object detections; no manual audit",
        "1,000 images / 14 classes; imbalanced, sparse classes",
        "One training run per architecture",
        "No geospatial metadata or geographic transfer test",
        "The supplied test split has already been inspected",
    ], 1.12, 2.65, 5.0, 2.55, size=13, gap=0.49)
    add_box(slide, 6.86, 1.9, 5.7, 3.85, MINT)
    add_text(slide, "NEXT STEPS", 7.16, 2.2, 4.8, 0.3,
             size=11, color=TEAL, bold=True)
    add_bullets(slide, [
        "Audit image–label pairs and near duplicates",
        "Repeat comparisons across multiple random seeds",
        "Collect a fresh test set with location / sensor provenance",
        "Choose any tuning only on train + validation data",
    ], 7.2, 2.65, 4.9, 2.55, size=13, gap=0.54)
    add_text(slide, "Do not interpret softmax confidence as a calibrated probability of correctness.",
             0.92, 6.14, 11.5, 0.35, size=13, color=NAVY, bold=True)

    # 9. Closing
    slide = pres.slides.add_slide(pres.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = rgb(NAVY)
    add_text(slide, "TAKE-HOME", 0.82, 0.82, 4.0, 0.3,
             size=11, color="A7DCD8", bold=True)
    add_text(slide, "A practical first look—not the final word.",
             0.82, 1.43, 10.7, 0.72, size=32, color=WHITE, bold=True)
    add_bullets(slide, [
        "SatScene turns an overhead RGB image into a scene prediction and ranked alternatives.",
        "ConvNeXtV2-Tiny performed best on the reported splits; the comparison is single-run.",
        "The interface is ready to demonstrate locally; GIS interpretation remains with the analyst.",
    ], 0.92, 2.7, 10.8, 2.4, size=18, color=WHITE, gap=0.76)
    add_box(slide, 0.88, 5.68, 11.4, 0.8, "214657")
    add_text(slide,
             "Paper source: paper/main.tex · Dataset, Methods, Results, Software/Reproducibility, Limitations",
             1.15, 5.96, 10.85, 0.26, size=11, color="DCE9F1")
    add_text(slide, "Thank you", 0.92, 6.88, 4.0, 0.28,
             size=13, color="A7DCD8", bold=True)
    add_text(slide, "09", 12.15, 7.08, 0.5, 0.25,
             size=9, color="A7DCD8", bold=True, align=PP_ALIGN.RIGHT)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pres.save(OUTPUT)
    print(f"Created {OUTPUT} ({len(pres.slides)} slides)")


if __name__ == "__main__":
    build_deck()
