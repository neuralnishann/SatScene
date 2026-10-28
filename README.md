<div align="center">

# 🛰️ SatScene: Land-Use Scene Classification with ConvNeXt-Tiny and a Rule-Based Decision-Support Agent

*Geospatial Data Science · Home Assignment 1 · Software Project · 2026*

</div>

---

## Abstract

SatScene is a remote-sensing image-classification project and reproducible software
workflow for assigning one of 14 land-use scene labels to satellite or aerial images.
The repository includes YOLO-annotation parsing, transfer-learning training and
evaluation, a browser interface, and an optional native macOS interface. A comparison
of three pretrained vision backbones on a 200-image validation split found that
SwinV2-Tiny achieved the highest top-1 accuracy (78.5%) and macro-F1 (0.766), while
ConvNeXtV2-Tiny achieved 93.5% top-3 accuracy. These are single-run validation results,
not estimates from the held-out test set. A separate saved ConvNeXt-Tiny test evaluation
scored 6% accuracy and predicted every test image as Agriculture, exposing a severe,
unresolved generalization failure. The results therefore support a software and
experimental comparison, not a claim of operationally reliable classification.

**Keywords:** remote sensing; satellite imagery; scene classification; transfer
learning; ConvNeXt; Swin Transformer; DINOv2; decision support

## Contents

- [1. Project scope](#1-project-scope)
- [2. Dataset and label construction](#2-dataset-and-label-construction)
- [3. Methods](#3-methods)
- [4. Experimental protocol](#4-experimental-protocol)
- [5. Results](#5-results)
- [6. Discussion](#6-discussion)
- [7. Limitations](#7-limitations)
- [8. Reproducibility and use](#8-reproducibility-and-use)
- [9. Software and archive contents](#9-software-and-archive-contents)
- [References and licenses](#references-and-licenses)

## 1. Project scope

SatScene studies image-level land-use scene classification using a small remote-sensing
dataset. It packages data handling, model training, validation, evaluation, and
single-image prediction in a Python project. The local interfaces show ranked class
predictions and rule-based follow-up suggestions for human review.

The supplied images are ordinary JPEGs without coordinates, a coordinate reference
system, geotransforms, or multispectral bands. SatScene therefore analyzes image
appearance only: it does not perform georeferenced raster analysis, calculate NDVI,
read elevation data, or execute the suggested follow-up analyses.

## 2. Dataset and label construction

The source is the [Remote Sensing Satellite Images dataset on Kaggle](https://www.kaggle.com/datasets/umeradnaan/remote-sensing-satellite-images),
distributed in a [Roboflow YOLOv8 export](https://universe.roboflow.com/image-captioning-for-remote-sensingdata-0zids/remote-sensing-data/dataset/2).
The export contains 1,000 images in supplied train, validation, and test splits
(700/200/100) and 14 classes:

> Agriculture · Airport · Beach · City · Desert · Forest · Grassland · Highway · Lake ·
> Mountain · Parking · Port · Railway · River

The export annotations are object detections rather than scene labels. For each image,
the implementation chooses the most frequent class ID in its YOLO annotation file;
ties go to the class appearing first. Images with missing or empty annotation files are
skipped. This provides a single-label classification target, but the dominant detected
object is only a proxy for the overall scene.

The training split is imbalanced: the recorded distribution ranges from 118 City images
to 16 River images (7.4:1). The dataset is not included in Git. The downloaded Roboflow
export identifies its license as CC BY 4.0; verify attribution and redistribution terms
at the [source](https://universe.roboflow.com/image-captioning-for-remote-sensingdata-0zids/remote-sensing-data/dataset/2)
before reuse. Dataset provenance is also recorded in
[`resources/dataset_source.txt`](resources/dataset_source.txt).

## 3. Methods

### 3.1 Image preparation

Images are read as RGB, resized to the model's configured square input size, and
normalized using ImageNet channel statistics. Training augmentation applies horizontal
flip, vertical flip, and random 90-degree rotation, each with probability 0.5. Validation
and evaluation use resize and normalization without random augmentation.

### 3.2 Model comparison

The archived comparison fine-tunes three pretrained `timm` classifiers. Each run uses
the same 14 classes, seed 42, five epochs, batch size 16, AdamW with learning rate
`3e-4` and weight decay `1e-4`, cosine learning-rate scheduling, and checkpoint
selection by validation macro-F1. Input size is 256 for SwinV2-Tiny and 224 for the
other two models.

| Model | `timm` model identifier | Input size |
| --- | --- | ---: |
| ConvNeXtV2-Tiny | `convnextv2_tiny.fcmae_ft_in1k` | 224 |
| SwinV2-Tiny | `swinv2_tiny_window8_256.ms_in1k` | 256 |
| DINOv2 ViT-S/14 | `vit_small_patch14_dinov2.lvd142m` | 224 |

### 3.3 Prediction and decision support

For an input image, the classifier returns softmax scores for the 14 classes. The
rule-based reporting layer presents the top predictions, flags low top-1 confidence,
and looks up a static suggestion associated with the predicted class. Suggestions are
for an analyst; the application does not carry them out.

## 4. Experimental protocol

The reported architecture comparison uses the validation split (`n = 200`) for model
selection and comparison. Its results are from one run and have no across-seed
uncertainty estimate. Accuracy intervals below are 95% Wilson score intervals; they
describe binomial sampling uncertainty for the validation sample, not variation from
retraining.

The test set is intended for final evaluation after the method is selected. The
separately archived test output is from the ConvNeXt-Tiny command-line pipeline, not
from the three comparison checkpoints. It is reported independently below and was not
used to select the comparison models.

## 5. Results

### 5.1 Validation comparison

| Model | Accuracy (95% Wilson CI) | Macro-F1 | Balanced accuracy | Weighted F1 | Top-3 accuracy | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **SwinV2-Tiny** | **78.5% (72.3–83.6%)** | **0.766** | **0.763** | **0.777** | 90.0% | **0.068** |
| ConvNeXtV2-Tiny | 76.5% (70.2–81.8%) | 0.743 | 0.736 | 0.764 | **93.5%** | 0.072 |
| DINOv2 ViT-S/14 | 32.0% (25.9–38.8%) | 0.222 | 0.246 | 0.269 | 55.5% | 0.085 |

The complete aggregate metrics are in
[`runs/model_comparison_metrics.csv`](runs/model_comparison_metrics.csv). Each model
folder contains its exact run configuration, per-image predictions, per-class report,
confusion matrices, and calibration diagnostics.

### 5.2 Model-level diagnostics

<details>
<summary><strong>SwinV2-Tiny</strong> — best validation accuracy and macro-F1</summary>

![SwinV2-Tiny validation diagnostics: confusion matrices, per-class F1, and calibration](runs/swinv2_tiny_window8/validation_diagnostics.png)

[Full metrics](runs/swinv2_tiny_window8/metrics_valid.json) ·
[Per-image predictions](runs/swinv2_tiny_window8/predictions_valid.csv) ·
[Run configuration](runs/swinv2_tiny_window8/config.json)

</details>

<details>
<summary><strong>ConvNeXtV2-Tiny</strong> — best top-3 accuracy</summary>

![ConvNeXtV2-Tiny validation diagnostics: confusion matrices, per-class F1, and calibration](runs/convnextv2_tiny/validation_diagnostics.png)

[Full metrics](runs/convnextv2_tiny/metrics_valid.json) ·
[Per-image predictions](runs/convnextv2_tiny/predictions_valid.csv) ·
[Run configuration](runs/convnextv2_tiny/config.json)

</details>

<details>
<summary><strong>DINOv2 ViT-S/14</strong></summary>

![DINOv2 ViT-S/14 validation diagnostics: confusion matrices, per-class F1, and calibration](runs/dinov2_vit_small_patch14/validation_diagnostics.png)

[Full metrics](runs/dinov2_vit_small_patch14/metrics_valid.json) ·
[Per-image predictions](runs/dinov2_vit_small_patch14/predictions_valid.csv) ·
[Run configuration](runs/dinov2_vit_small_patch14/config.json)

</details>

### 5.3 Per-class performance of SwinV2-Tiny

The per-class F1 results show why the macro-average and class support matter. SwinV2
performed strongly on several classes, while River and Grassland remained difficult.

| Class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Agriculture | 15 | 0.800 | 0.800 | 0.800 |
| Airport | 12 | 0.714 | 0.833 | 0.769 |
| Beach | 9 | 0.889 | 0.889 | 0.889 |
| City | 37 | 0.761 | 0.946 | 0.843 |
| Desert | 16 | 0.929 | 0.813 | 0.867 |
| Forest | 10 | 0.412 | 0.700 | 0.519 |
| Grassland | 21 | 0.667 | 0.381 | 0.485 |
| Highway | 11 | 0.818 | 0.818 | 0.818 |
| Lake | 10 | 0.750 | 0.900 | 0.818 |
| Mountain | 19 | 0.895 | 0.895 | 0.895 |
| Parking | 14 | 0.933 | 1.000 | 0.966 |
| Port | 9 | 1.000 | 0.889 | 0.941 |
| Railway | 9 | 1.000 | 0.444 | 0.615 |
| River | 8 | 0.750 | 0.375 | 0.500 |

Values are transcribed from
[`runs/swinv2_tiny_window8/metrics_valid.json`](runs/swinv2_tiny_window8/metrics_valid.json);
per-class scores are sensitive to the small number of examples in each class.

### 5.4 Comparison figures

![Validation metrics for the best checkpoint from each model](runs/model_comparison_score_metrics.png)

![Validation accuracy over the comparison training epochs](runs/model_comparison_validation_accuracy.png)

![Training loss over the comparison training epochs](runs/model_comparison_training_loss.png)

### 5.5 Separate held-out test result

The archived test evaluation for the packaged ConvNeXt-Tiny pipeline scored **6%
accuracy (6/100)** and **0.008 macro-F1**. All 100 test images were predicted as
Agriculture, with a top-label expected calibration error of 0.940. This severe
validation/test discrepancy is unresolved. The result is a warning against interpreting
the validation comparison as evidence of generalization or deployment readiness.

The recorded output is available in
[`iodata/sample_output/metrics_test.json`](iodata/sample_output/metrics_test.json), with
the example prediction in [`iodata/sample_output/prediction.csv`](iodata/sample_output/prediction.csv).

## 6. Discussion

Within this single validation comparison, SwinV2-Tiny ranked first in top-1 accuracy,
macro-F1, balanced accuracy, and weighted F1. ConvNeXtV2-Tiny had the highest top-3
accuracy, which may be useful when predictions are reviewed by a person, but does not
replace top-1 or per-class performance. DINOv2 ViT-S/14 performed substantially worse
under the recorded training setup.

The per-class report shows high validation F1 for Parking, Port, Beach, and Mountain,
but lower F1 for Grassland, Forest, and River. Small class supports make these estimates
uncertain. The separate test collapse also shows that good validation metrics alone do
not establish robust performance. Further investigation should start with annotation
quality, split integrity, class ordering, checkpoint loading, and per-class test
predictions before any deployment claim.

## 7. Limitations

- **Severe generalization failure:** the available ConvNeXt-Tiny test output is 6%
  accurate and predicts a single class for every sample. The cause has not been
  established.
- **Single comparison run:** there are no repeated seeds or variance estimates.
- **Small, imbalanced dataset:** 1,000 images are few for a 14-class task, and class
  supports range from 8 to 37 in validation.
- **Proxy labels:** assigning the most frequent detection class as the scene label can
  misrepresent mixed or multi-object scenes.
- **No geospatial metadata:** spatial splits, geographic generalization, and actual GIS
  measurements cannot be evaluated with these JPEGs.
- **Dataset and model provenance:** dataset redistribution and pretrained-weight terms
  are external to this code license and must be checked independently.
- **Validation-selected results:** the reported comparison values are model-selection
  metrics, not final test estimates.

The earlier notebook also contains historical output that is not used as a verified
result because its saved label-generation code and displayed class-distribution output
disagree. The current comparison values above are taken from the three archived run
folders.

## 8. Reproducibility and use

### 8.1 Environment

Requires Python 3.10 or later; Python 3.11 is recommended. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

### 8.2 Dataset setup

The dataset is not tracked in the repository. Configure Kaggle CLI credentials, then
download and extract it:

```bash
bash scripts/download_data.sh
```

The configured directory is `data/Remote Sensing Data.v2i.yolov8`, containing
`data.yaml` and the `train`, `valid`, and `test` split directories. Change `data.root`
in [`configs/default.yaml`](configs/default.yaml) if your extracted path differs.

### 8.3 Train and evaluate the packaged ConvNeXt-Tiny pipeline

```bash
python -m satscene.train --config configs/default.yaml
python -m satscene.evaluate --config configs/default.yaml --split test
```

Training writes the selected checkpoint and history to `runs/convnext_tiny/`. Evaluation
writes aggregate metrics, predictions, and diagnostic plots in the configured output
directory. Reserve the test split for final evaluation after model selection.

### 8.4 Reproduce the three-model comparison

Open [`notebooks/GIS_project_original.ipynb`](notebooks/GIS_project_original.ipynb)
in Jupyter or Google Colab and run the comparison workflow. It fine-tunes the three
models described in [Section 3](#3-methods) and writes per-model reports and plots under
`runs/`. The large `.pth` checkpoint files are deliberately excluded from Git; rerun
the notebook to recreate them.

### 8.5 Run the applications and tests

```bash
# Local browser interface; open http://127.0.0.1:7860
python -m satscene.app --config configs/default.yaml

# Optional native macOS interface
python -m satscene.desktop --config configs/default.yaml

# Tests
python -m pip install -e '.[dev]'
python -m pytest -q
```

On Apple Silicon, PyTorch uses MPS when available and otherwise falls back to CPU.
CUDA is used when available. Pretrained weights are downloaded on the first run, so
initial training requires internet access. Checkpoints are ignored by Git because of
their size; supply a locally available checkpoint using `--checkpoint` where needed.

## 9. Software and archive contents

| Path | Contents |
| --- | --- |
| `src/satscene/labels.py` | YOLO label parsing and class utilities |
| `src/satscene/data.py` | Image transforms and data loaders |
| `src/satscene/model.py` | Model construction and checkpoint utilities |
| `src/satscene/train.py` | Configurable training pipeline |
| `src/satscene/evaluate.py` | Evaluation metrics, predictions, and diagnostic plots |
| `src/satscene/agent.py`, `knowledge.py` | Prediction report and static class suggestions |
| `src/satscene/app.py`, `desktop.py` | Browser and optional macOS interfaces |
| `configs/default.yaml` | Packaged ConvNeXt-Tiny configuration |
| `notebooks/` | Three-model comparison workflow |
| `runs/` | Comparison metrics, predictions, configurations, and plots |
| `iodata/` | Sample input and archived test-output example |
| `docs/` | macOS setup and experiment protocol |
| `tests/` | Automated tests |
| `resources/` | Dataset provenance and references |

The comparison summaries and diagnostics are committed; trained checkpoint files are
not. The optional LaTeX/PDF submission materials in `paper/` are local-only and are not
part of the tracked software archive.

## References and licenses

- Liu et al. (2022). [A ConvNet for the 2020s](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html). CVPR.
- Liu et al. (2022). [Swin Transformer V2: Scaling Up Capacity and Resolution](https://arxiv.org/abs/2111.09883).
- Oquab et al. (2023). [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193).
- Wightman et al. [PyTorch Image Models (`timm`)](https://github.com/huggingface/pytorch-image-models).
- Dataset provenance and export details: [`resources/README.md`](resources/README.md).

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). The software is
licensed under the [MIT License](LICENSE). Dataset and pretrained-model licenses are
separate; consult their original sources before redistribution.
