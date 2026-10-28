# SatScene

Satellite and aerial image scene classification across 14 land-use classes. SatScene
includes a PyTorch training and evaluation pipeline, a local browser interface, and an
optional native macOS app. The interface displays the top predictions and static,
class-specific follow-up suggestions; it does not perform GIS analysis.

## Results

The model-comparison run evaluated three fine-tuned architectures on the same
200-image validation split. **SwinV2-Tiny led on validation accuracy and macro-F1**;
ConvNeXtV2-Tiny had the strongest top-3 accuracy.

| Model | Accuracy | Macro-F1 | Balanced accuracy | Top-3 accuracy |
| --- | ---: | ---: | ---: | ---: |
| **SwinV2-Tiny** | **78.5%** | **0.766** | **0.763** | 90.0% |
| ConvNeXtV2-Tiny | 76.5% | 0.743 | 0.736 | **93.5%** |
| DINOv2 ViT-S/14 | 32.0% | 0.222 | 0.246 | 55.5% |

The best run reached 78.5% validation accuracy (95% Wilson interval: 72.3–83.6%)
and 0.766 macro-F1 across 14 classes. See the [complete comparison metrics](runs/model_comparison_metrics.csv)
and per-model reports for [SwinV2-Tiny](runs/swinv2_tiny_window8/metrics_valid.json),
[ConvNeXtV2-Tiny](runs/convnextv2_tiny/metrics_valid.json), and
[DINOv2 ViT-S/14](runs/dinov2_vit_small_patch14/metrics_valid.json).

Each model folder also contains its validation predictions, class-level scores, and
diagnostic plots:

<details>
<summary><strong>SwinV2-Tiny</strong> — 78.5% accuracy · 0.766 macro-F1 · 0.900 top-3 accuracy</summary>

![SwinV2-Tiny validation diagnostics: confusion matrices, per-class F1, and calibration](runs/swinv2_tiny_window8/validation_diagnostics.png)

Files: [metrics](runs/swinv2_tiny_window8/metrics_valid.json) ·
[predictions](runs/swinv2_tiny_window8/predictions_valid.csv) ·
[configuration](runs/swinv2_tiny_window8/config.json)

</details>

<details>
<summary><strong>ConvNeXtV2-Tiny</strong> — 76.5% accuracy · 0.743 macro-F1 · 0.935 top-3 accuracy</summary>

![ConvNeXtV2-Tiny validation diagnostics: confusion matrices, per-class F1, and calibration](runs/convnextv2_tiny/validation_diagnostics.png)

Files: [metrics](runs/convnextv2_tiny/metrics_valid.json) ·
[predictions](runs/convnextv2_tiny/predictions_valid.csv) ·
[configuration](runs/convnextv2_tiny/config.json)

</details>

<details>
<summary><strong>DINOv2 ViT-S/14</strong> — 32.0% accuracy · 0.222 macro-F1 · 0.555 top-3 accuracy</summary>

![DINOv2 ViT-S/14 validation diagnostics: confusion matrices, per-class F1, and calibration](runs/dinov2_vit_small_patch14/validation_diagnostics.png)

Files: [metrics](runs/dinov2_vit_small_patch14/metrics_valid.json) ·
[predictions](runs/dinov2_vit_small_patch14/predictions_valid.csv) ·
[configuration](runs/dinov2_vit_small_patch14/config.json)

</details>

### Comparison plots

![Validation metrics for the best checkpoint from each model](runs/model_comparison_score_metrics.png)

![Validation accuracy over the comparison training epochs](runs/model_comparison_validation_accuracy.png)

![Training loss over the comparison training epochs](runs/model_comparison_training_loss.png)

These are validation results from a single comparison run, not held-out test results.
Separately, the sample held-out evaluation in
[`iodata/sample_output/metrics_test.json`](iodata/sample_output/metrics_test.json) is a
failure case from a ConvNeXt-Tiny run: 6% accuracy on 100 images, with every image
predicted as Agriculture. The comparison results do not establish test performance or
operational suitability; the test failure remains unresolved.

## Dataset

The project uses the [Remote Sensing Satellite Images dataset](https://www.kaggle.com/datasets/umeradnaan/remote-sensing-satellite-images),
distributed as a YOLOv8 detection export. It contains 14 classes: Agriculture, Airport,
Beach, City, Desert, Forest, Grassland, Highway, Lake, Mountain, Parking, Port, Railway,
and River.

The training pipeline converts detection annotations into one scene label per image by
choosing the most frequent class in its annotation file. Images with missing or empty
annotations are skipped. The images are JPEGs without geospatial coordinates or
multispectral bands, so this project classifies image content; it is not georeferenced
raster analysis.

The dataset is not included in Git. Its downloaded export reports a CC BY 4.0 license;
check the [source export](https://universe.roboflow.com/image-captioning-for-remote-sensingdata-0zids/remote-sensing-data/dataset/2)
for current license and attribution terms before redistribution. See
[`resources/dataset_source.txt`](resources/dataset_source.txt).

## Setup

Requires Python 3.10 or later (Python 3.11 is recommended).

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Download and extract the dataset with the Kaggle CLI:

```bash
bash scripts/download_data.sh
```

Configure Kaggle credentials before downloading. The training configuration expects
`data/Remote Sensing Data.v2i.yolov8` to contain `data.yaml` and the `train`, `valid`,
and `test` folders. If the dataset was extracted elsewhere, update `data.root` in
[`configs/default.yaml`](configs/default.yaml).

## Train, evaluate, and run

Train the configured ConvNeXt-Tiny model. The best validation checkpoint and training
history are written under `runs/convnext_tiny/`.

```bash
python -m satscene.train --config configs/default.yaml
```

Evaluate the selected checkpoint on the held-out test split after model selection:

```bash
python -m satscene.evaluate --config configs/default.yaml --split test
```

Run the local browser app:

```bash
python -m satscene.app --config configs/default.yaml
```

Open `http://127.0.0.1:7860`. To use another checkpoint or port, pass `--checkpoint`
and/or `--port`. The optional native macOS app is:

```bash
python -m satscene.desktop --config configs/default.yaml
```

Model checkpoints are intentionally excluded from Git because of their size. Train a
checkpoint locally before launching the app, or provide a compatible checkpoint with
`--checkpoint`. For Apple Silicon, PyTorch uses MPS when available; otherwise it falls
back to CPU. First-time training downloads pretrained weights and requires internet
access.

Run tests with:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

## Model comparison

The notebook [`notebooks/GIS_project_original.ipynb`](notebooks/GIS_project_original.ipynb)
contains the ConvNeXtV2-Tiny, SwinV2-Tiny, and DINOv2 ViT-S/14 comparison workflow. Its
validation summaries and plots are stored in `runs/`. Checkpoint files are not tracked;
rerun the notebook to generate them. The standard command-line training pipeline above
uses the model and settings in `configs/default.yaml`.

## Project layout

```text
configs/       Training configuration
docs/          Setup and experiment notes
iodata/        Small example input and evaluation output
notebooks/     Model-comparison notebook
resources/     Dataset attribution and references
runs/          Committed metrics and plots; checkpoints are excluded
scripts/       Dataset download and plotting utilities
src/satscene/  Training, evaluation, prediction, and app code
tests/         Automated tests
```

See [`docs/MAC_SETUP.md`](docs/MAC_SETUP.md) for a macOS walkthrough and
[`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md) for the evaluation protocol.

## Citation and license

Citation metadata is in [`CITATION.cff`](CITATION.cff). The project code is licensed
under the [MIT License](LICENSE). Dataset and pretrained-model terms are separate; see
their respective sources.
