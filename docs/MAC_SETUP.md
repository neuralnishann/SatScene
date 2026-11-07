# SatScene on macOS

The browser interface is the recommended local application. An optional native Cocoa
window is also available. Both accept a trained checkpoint. On Apple Silicon, PyTorch
uses MPS when supported by the installed build; otherwise it uses the CPU. Intel Macs
use the CPU. First-time training downloads pretrained model weights and requires an
internet connection.

## 1. Install Python and project dependencies

Install Python 3.11, then run these commands from the project folder:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Keep the `.venv` environment active when running SatScene. The native desktop app also
needs the PyObjC dependency included in `requirements.txt`.

## 2. Download the dataset

Configure Kaggle credentials in `~/.kaggle/kaggle.json` or set `KAGGLE_USERNAME` and
`KAGGLE_KEY`, then download the dataset:

```bash
bash scripts/download_data.sh
```

The expected root is `data/Remote Sensing Data.v2i.yolov8`, containing `data.yaml` and
`train`, `valid`, and `test` directories with `images/` and `labels/` subdirectories.
If the extracted directory has another name, change `data.root` in the configuration
you use to point to the directory containing `data.yaml`.

## 3. Final three-model comparison

The archived comparison used the supplied 700/200 train/validation split, seed 42,
10 epochs, batch size 16, AdamW with learning rate `3e-4` and weight decay `1e-4`,
train-only horizontal and vertical flips and 90-degree rotations (each probability
0.5), and best-checkpoint selection by validation macro-F1. The 100-image test split
was not used for this comparison. These are single-run validation results, not
repeated-seed estimates:

| Model | Input size | Best epoch | Validation accuracy | Macro-F1 | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| ConvNeXtV2-Tiny (`convnextv2_tiny.fcmae_ft_in1k`) | 224 | 9 | 80.0% (160/200) | 0.7881 | 0.7876 |
| SwinV2-Tiny (`swinv2_tiny_window8_256.ms_in1k`) | 256 | 10 | 79.5% (159/200) | 0.7817 | 0.7770 |
| DINOv2 ViT-S/14 (`vit_small_patch14_dinov2.lvd142m`) | 224 | 10 | 38.0% (76/200) | 0.3049 | 0.3441 |

Detailed metrics and diagnostics are in [`../runs/README.md`](../runs/README.md). The
three trained checkpoints are present in this workspace and their model metadata
matches their run configurations. Each is ignored by Git due to its size, so a fresh
clone may not contain them. Their paths are:

```text
runs/convnextv2_tiny/best_model.pth
runs/swinv2_tiny_window8/best_model.pth
runs/dinov2_vit_small_patch14/best_model.pth
```

The repository's `configs/default.yaml` is a separate ConvNeXt-Tiny packaged training
configuration. Its `runs/convnext_tiny/` output directory and checkpoint are not
present in this workspace. Running its training command below would produce a
separate model and would not reproduce these three comparison runs or their metrics.

## 4. Train the packaged ConvNeXt-Tiny configuration

```bash
python -m satscene.train --config configs/default.yaml
```

This trains `convnext_tiny.fb_in22k_ft_in1k` and writes its best checkpoint to
`runs/convnext_tiny/best_model.pth`. Training needs the local dataset and may download
pretrained weights on first use. CPU training can take a long time. This model is
separate from the three checkpoints already present under `runs/`.

## 5. Launch an application with a checkpoint

For the strongest result in the saved comparison, launch the available ConvNeXtV2-Tiny
checkpoint:

```bash
python -m satscene.app --config configs/default.yaml \
  --checkpoint runs/convnextv2_tiny/best_model.pth
```

To use SwinV2-Tiny or DINOv2, replace the path with the corresponding checkpoint from
Section 3. The browser opens at `http://127.0.0.1:7860`. Upload an image and select
**Analyze image** to see ranked predictions, confidence, and suggested follow-up.
**Clear all** resets the image and report. Pass `--port 7861` to use another port;
`--share` creates a public Gradio link. Press Control-C in Terminal to stop the app.

To use the native macOS app with the same checkpoint, run:

```bash
python -m satscene.desktop --config configs/default.yaml \
  --checkpoint runs/convnextv2_tiny/best_model.pth
```

```bash
python -m satscene.desktop --config configs/default.yaml \
  --checkpoint runs/convnextv2_tiny/best_model.pth
```

```bash
python -m satscene.desktop --config configs/default.yaml \
  --checkpoint runs/convnextv2_tiny/best_model.pth
```

Change the checkpoint path to use either of the other comparison models. The desktop
app accepts an image by file picker or drag and drop. Both interfaces load model
architecture and image size from checkpoint metadata.

To compare all three saved models on one uploaded image in the browser app, run:

```bash
python -m satscene.app --config configs/default.yaml \
  --compare-checkpoints \
  runs/convnextv2_tiny/best_model.pth \
  runs/swinv2_tiny_window8/best_model.pth \
  runs/dinov2_vit_small_patch14/best_model.pth
```

Upload an image once to see each model's confidence-ranked predictions side by side.
The app loads all three checkpoints and analyzes the image with each model in turn.
Use `--compare-checkpoints` with exactly three checkpoint paths; it cannot be combined
with `--checkpoint`.

To compare the same three models in the native macOS desktop app, run:

```bash
python -m satscene.desktop --config configs/default.yaml \
  --compare-checkpoints \
  runs/convnextv2_tiny/best_model.pth \
  runs/swinv2_tiny_window8/best_model.pth \
  runs/dinov2_vit_small_patch14/best_model.pth
```

Choose or drag in one image and the three models automatically show their top
predictions, confidence, and ranked alternatives side by side. Inference runs in the
background so the desktop UI remains responsive. The existing `--checkpoint` option
continues to launch a single model, which still requires selecting **Analyze image**.

## 6. Evaluate a held-out test split

Only evaluate after model selection is finished:

```bash
python -m satscene.evaluate --config configs/default.yaml --split test \
  --checkpoints \
  runs/convnextv2_tiny/best_model.pth \
  runs/swinv2_tiny_window8/best_model.pth \
  runs/dinov2_vit_small_patch14/best_model.pth
```

This evaluates the same held-out split with all three checkpoints in one run. Each
model's `metrics_test.json`, `predictions_test.csv`, and `confusion_test.png` are saved
beside its checkpoint under `runs/`. For a single checkpoint, `--output-dir` can still
be used to choose the output location. Keep test results separate from model selection.
The existing files in `iodata/sample_output/` record 86% accuracy, macro-F1 0.783, and
balanced accuracy 0.777 on 100 test images. They do not record the model/checkpoint
identity and are not test metrics for the archived three-model comparison. The test
split has already been evaluated, so it is not a blind final assessment.

## Troubleshooting

- **Checkpoint not found:** use one of the exact paths listed in Section 3 and confirm
  the `.pth` file is present. The default command without `--checkpoint` looks for the
  separate `runs/convnext_tiny/best_model.pth` and will fail until that model is trained.
- **No labeled images found:** check `data.root`, split folder names, and YOLO label
  text files beside the images under each split's `labels/` directory.
- **MPS unavailable:** PyTorch falls back to CPU. Check with
  `python -c 'import torch; print(torch.backends.mps.is_available())'`.
- **Port 7860 is occupied:** pass `--port 7861` to the browser app.
- **Kaggle download fails:** verify the credentials and dataset access.

The supplied images are non-georeferenced JPEGs. SatScene classifies image content; it
does not read coordinates or run NDVI, DEM, or other GIS analyses.
