<div align="center">

# 🛰️ SatScene: Land-Use Scene Classification with ConvNeXt-Tiny and a Rule-Based Decision-Support Agent

![python](https://img.shields.io/badge/python-3.9%2B-blue)
![pytorch](https://img.shields.io/badge/PyTorch-timm-ee4c2c)
![license](https://img.shields.io/badge/license-MIT-green)

*Geospatial Data Science course · Home Assignment 1 (Software Project) · 2026*

</div>

---

## Abstract

We fine-tune ImageNet-pretrained **ConvNeXt-Tiny** on a 14-class overhead-imagery dataset and provide a local image-classification GUI with rule-based follow-up suggestions. A packaged seeded run selected by validation macro-F1 reached **77.5 % validation accuracy** and **0.761 validation macro-F1**. Its saved held-out test evaluation failed badly: **6 % accuracy**, **0.008 macro-F1**, and every test image was predicted as Agriculture with near-certain confidence. This project is a working software implementation, but the classifier is not currently suitable for use. The optional LaTeX report is in the local `paper/` folder; it is ignored by Git and must be included manually in the submission archive.

> **Read this first — status of the numbers.** The 80.5 % result below is historical notebook output and is unverified because saved notebook code and output disagree. The packaged pipeline has been run locally: `runs/convnext_tiny/history.json` records a seeded run with 77.5 % validation accuracy and 0.761 validation macro-F1. `runs/convnext_tiny/metrics_test.json` records a test failure (6 % accuracy; all 100 predictions are Agriculture). Do not describe the held-out result as successful. See [Known limitations](#known-limitations).

## Results at a glance

| | Value |
| --- | --- |
| Backbone | `convnext_tiny.fb_in22k_ft_in1k` (timm), 27.83 M params |
| Data | 14 classes · 700 train / 200 val / 100 test |
| Historical notebook checkpoint | epoch 7 of 10 (unverified) |
| Packaged checkpoint | epoch 4 of 10 |
| Historical notebook validation accuracy | 80.5 % (unverified notebook output) |
| Packaged run validation accuracy | 77.5 % (155/200; 95 % Wilson CI 71.2–82.7 %) |
| Packaged run validation macro-F1 | 0.761 (checkpoint selected by macro-F1) |
| **Saved held-out test accuracy** | **6 % (6/100)**; every prediction is Agriculture |

<p align="center">
  <img src="results/figures/training_curves.png" width="760" alt="Training loss and accuracy curves">
</p>

<p align="center">
  <img src="results/figures/class_distribution.png" width="760" alt="Training-set class distribution">
</p>

Caveats: this is one training run; the test performance collapses despite reasonable validation metrics. The cause is not established. The test result has not been used to tune the model.

## Method in brief

| Component | Choice |
| --- | --- |
| Input | RGB, resized to 224 × 224, ImageNet mean/std |
| Augmentation | horizontal flip, vertical flip, random 90° rotation (p = 0.5 each) |
| Loss / optimiser | cross-entropy · AdamW (lr 3e-4, wd 1e-4) |
| Schedule | 10 epochs, batch 32, cosine annealing (`T_max = 10`) |
| Precision | fp16 mixed precision with loss scaling (CUDA only) |
| Model selection | new packaged runs select by validation macro-F1 (configurable); historical notebook checkpoint used accuracy |

**Agent layer** (`src/satscene/agent.py`): (1) *perceive* — softmax over the 14 classes; (2) *reason* — look up the predicted class in a static knowledge table ([`knowledge.py`](src/satscene/knowledge.py)); (3) *report* — markdown with top-3 classes, a suggested follow-up analysis, and a low-confidence flag (< 0.5). It is a rule-based decision-support layer: the suggestions are for a human analyst and nothing is executed.

## Dataset

[`umeradnaan/remote-sensing-satellite-images`](https://www.kaggle.com/datasets/umeradnaan/remote-sensing-satellite-images) on Kaggle (CC0-1.0), a Roboflow **YOLOv8 detection** export with 14 classes: *Agriculture, Airport, Beach, City, Desert, Forest, Grassland, Highway, Lake, Mountain, Parking, Port, Railway, River*.

Scene labels are **derived** from the detection labels: each image gets its most frequent class id (ties → first appearing); images with empty label files are skipped and reported. The images are plain JPEGs with **no CRS, geotransform or band metadata**, so this is image classification rather than georeferenced raster analysis.

## Installation

```bash
# From the project directory:
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .            # makes `python -m satscene.*` available
```

On Apple Silicon, PyTorch uses the MPS backend automatically when available; Intel Macs
and other machines use CPU. CUDA is used when available. Python 3.11 is the recommended
version for the dependency set. See [the Mac setup guide](docs/MAC_SETUP.md) for a clean
installation and first-run walkthrough.

## Reproducing the results

### Compare recent vision models in Google Colab

Open [`notebooks/GIS_project_original.ipynb`](notebooks/GIS_project_original.ipynb) in Colab or local Jupyter and run its model-comparison workflow. Keep the project source (`src/satscene/`) available; in Colab, the notebook mounts Drive and checks `MyDrive/SatScene`. Step 1 searches common `data/`, `dataset/`, and `datasets/` paths for an existing YOLO export, then downloads [`umeradnaan/remote-sensing-satellite-images`](https://www.kaggle.com/datasets/umeradnaan/remote-sensing-satellite-images) into the project if needed. Kaggle credentials may be required: add a `KAGGLE_API_TOKEN` Colab secret or configure the Kaggle CLI locally. Set `SATSCENE_DATASET_DIR` to use a dataset in a custom location.

On Apple Silicon, Step 2 selects MPS when PyTorch reports it available, uses batch size 8 and zero loader workers by default, and clears MPS cache between models. Lower the batch size to 4 if memory pressure occurs. Check the printed device before training; it should say `mps` on the Mac. The notebook fine-tunes ConvNeXtV2-Tiny, SwinV2-Tiny, and DINOv2 ViT-S/14 sequentially. Each model saves its best validation-macro-F1 weights and its own configuration, history, and metrics:

- `runs/convnextv2_tiny/best_model.pth`
- `runs/swinv2_tiny_window8/best_model.pth`
- `runs/dinov2_vit_small_patch14/best_model.pth`

The notebook saves one cross-model training-loss graph, one cross-model validation-accuracy graph, full validation reports, and a ZIP of the three runs. It does not use the test split for model selection.

Run these commands in Terminal from the project root. The module commands (`python -m …`)
are important: do not launch `src/satscene/app.py` as a file, because that bypasses the
installed `satscene` package imports.

```bash
# 1. Open the project and activate its environment
cd "/path/to/SatScene"
source .venv/bin/activate

# 2. Install dependencies and the local package (first setup, or after dependency changes)
python -m pip install -r requirements.txt
python -m pip install -e .

# 3. Data is already in data/ in this checkout; download only if data.yaml is absent.
#    Kaggle credentials required for download: ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY
if [ ! -f "data/Remote Sensing Data.v2i.yolov8/data.yaml" ]; then bash scripts/download_data.sh; fi

# 4. Train a fresh checkpoint (uses MPS on supported Apple Silicon Macs)
python -m satscene.train --config configs/default.yaml
#    -> runs/convnext_tiny/best_model.pth and history.json

# 5. Make training curves
python scripts/plot_training_history.py runs/convnext_tiny/history.json
#    -> runs/convnext_tiny/training_curves.png

# 6. Evaluate the selected checkpoint ONCE on the held-out test set
python -m satscene.evaluate --config configs/default.yaml --split test \
  --checkpoint runs/convnext_tiny/best_model.pth
#    -> metrics_test.json, predictions_test.csv, and confusion_test.png under runs/convnext_tiny/

# 7. Run the browser GUI at http://127.0.0.1:7860
python -m satscene.app --config configs/default.yaml \
  --checkpoint runs/convnext_tiny/best_model.pth

# Optional: run the native macOS desktop app
python -m satscene.desktop --config configs/default.yaml \
  --checkpoint runs/convnext_tiny/best_model.pth

# Optional: run the repository tests (install with `python -m pip install -e '.[dev]'` first)
python -m pytest -q
```

The saved packaged checkpoint at `runs/convnext_tiny/best_model.pth` also predicts
Agriculture for every image in the held-out test split despite 77.5 % validation accuracy.
Treat this as an unresolved generalization failure. If the data is already present, skip
the download step.

For plots from a new packaged training run, use `python scripts/plot_training_history.py runs/convnext_tiny/history.json`. Final evaluation writes metrics, per-image predictions, and a multi-panel confusion, per-class F1, and calibration figure. See [the experiment plan](docs/EXPERIMENT_PLAN.md) for a step-by-step protocol, metric definitions, and a seven-minute presentation outline.

Expect results to differ from the logged run: the notebook run was unseeded, and its saved label-derivation code conflicts with its output. The packaged training code is seeded and derives labels from the YOLO annotations. Newly trained checkpoints also store the class order, model name, and input size, so the GUI can use them without the dataset being present. After obtaining a new run, update `results/training_log.json` if you want the figures to reflect it, then run `python scripts/make_figures.py`.

## Repository structure

```
satellite-scene-agent/
├── README.md
├── .gitignore
├── CITATION.cff
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── src/satscene/
│   ├── labels.py            # YOLO annotation parsing and class utilities
│   ├── data.py              # image transforms and data loaders
│   ├── model.py             # ConvNeXt-Tiny and checkpoint utilities
│   ├── train.py             # training loop
│   ├── evaluate.py          # metrics, predictions, confusion and calibration plots
│   ├── agent.py             # prediction report wrapper
│   ├── knowledge.py         # class-specific follow-up suggestions
│   ├── metrics.py           # Wilson confidence interval
│   ├── app.py               # Gradio browser demo
│   └── desktop.py           # native macOS Cocoa desktop GUI
├── configs/default.yaml
├── scripts/
│   ├── download_data.sh
│   ├── make_figures.py
│   └── plot_training_history.py
├── docs/
│   ├── MAC_SETUP.md
│   └── EXPERIMENT_PLAN.md
├── results/
│   ├── training_log.json
│   └── figures/
├── notebooks/GIS_project_original.ipynb  # original code with explanatory Markdown added
├── tests/
│   ├── test_labels.py
│   └── test_knowledge_and_log.py
└── paper/                   # local only; ignored by Git
```

## What changed relative to the original notebook

| Notebook | This repo | Why |
| --- | --- | --- |
| Data cell assigns `class_names[0]` to **every** image ("dummy labels"), yet its saved output says *"Parsed true classes"* with a 14-class distribution | Real label parsing in `labels.py`, with unit tests | The saved code and saved output disagree; the logged results must have come from a different version of that cell |
| Agent knowledge table keyed on `"agricultural"`, `"terminal"`, `"car parking"`, `"road"`, `"train"`, `"domestic"` … | Keys equal the 14 dataset class names; a test asserts full coverage | 6 of 14 classes (Agriculture, Airport, Grassland, Highway, Parking, Railway) would have returned *"Unknown Land Use"*; Grassland had no entry |
| `GradScaler('cuda')` / `device_type='cuda'` hard-coded | Device-aware AMP | Crashes on CPU otherwise |
| No random seed | `seed` in config | Reproducibility |
| Test split loaded but never used | `evaluate.py` | Needed for an honest estimate |
| Single `analyze()` returning a string | Also `analyze_structured()` with top-3 and a low-confidence flag | Programmatic use |
| Unused imports (`seaborn`, `confusion_matrix`, …) | Used in `evaluate.py` | — |

The notebook code cells are preserved; explanatory Markdown has been added before each code cell. The notes call out the dummy-label, CUDA-only AMP, and class-name lookup limitations in the original code.

## Known limitations

- **Severe validation/test gap:** packaged evaluation scored 77.5 % validation accuracy but only 6 % test accuracy; every test prediction is Agriculture. The cause has not been established.
- **Historical notebook result is unverified:** its saved code assigns dummy labels while saved output claims 14-class labels.
- **Validation-selected checkpoint**: 77.5 % is a model-selection result, not an unbiased final estimate; the held-out test result is much worse.
- **Single packaged run**, so no variance across seeds. The validation confidence interval reflects sampling uncertainty only.
- **Over-fitting**: packaged training accuracy reaches 89.6 % by its best epoch vs 77.5 % validation accuracy; test performance is substantially worse.
- **Class imbalance** (City 118 vs River 16, 7.4 : 1). The original logged run has no per-class metrics; the packaged evaluator now reports them. Weighted loss is available via `class_weighted_loss: true` but still needs an ablation run.
- **Derived labels** from detection boxes may be noisy for mixed scenes; no check for near-duplicates across splits; sensor, resolution and dates unknown.
- **Not georeferenced**: no CRS or spectral bands, so no spatial cross-validation and no real NDVI. The agent's follow-ups are suggestions only.

## Roadmap

- [x] Run the packaged pipeline and evaluate once on the test split; the result is a severe all-Agriculture failure and must be investigated without tuning on test data
- [ ] Multi-seed runs (mean ± std)
- [ ] Ablations: class-weighted loss, stronger augmentation, label smoothing, TTA
- [ ] Near-duplicate audit across splits
- [ ] Georeferenced benchmark (e.g. EuroSAT) to connect the agent to real NDVI/DEM data

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). Add a repository URL before publishing or archiving the project.

## References

Key works are cited in the optional paper source under the local `paper/` folder. That folder is ignored by Git, so its bibliography is not present in repository clones.

## License

Code: [MIT](LICENSE). Dataset: CC0-1.0 (see its Kaggle page). Pre-trained weights are subject to their own licences via `timm`.
