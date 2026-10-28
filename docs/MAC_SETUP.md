# SatScene on macOS

SatScene's primary interface is a local browser app with a responsive upload and results
layout; the native macOS desktop app is optional.
On Apple Silicon, PyTorch uses MPS when it is available; otherwise inference and training
run on the CPU. First-time training downloads pretrained weights and needs internet access.

## 1. Install Python and create an environment

Install Python 3.11 (for example from python.org), then open Terminal and run these
commands from the SatScene project folder:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

If `python3.11` is not found, install Python 3.11 and reopen Terminal. Keep using the
`.venv` environment whenever you run the app.

## 2. Download and check the dataset

The download script uses the Kaggle CLI. Configure Kaggle API credentials in
`~/.kaggle/kaggle.json` or set `KAGGLE_USERNAME` and `KAGGLE_KEY`, then run:

```bash
bash scripts/download_data.sh
```

The configured data root is `data/Remote Sensing Data.v2i.yolov8`. It should contain
`data.yaml` and `train`, `valid`, and `test` folders, each with `images/` and `labels/`.
If Kaggle extracts to a differently named directory, update `data.root` in
`configs/default.yaml` to the directory containing `data.yaml`.

## 3. Train a model

```bash
python -m satscene.train --config configs/default.yaml
```

The best checkpoint is written to `runs/convnext_tiny/best_model.pth`. On Apple Silicon,
training uses MPS if supported by the installed PyTorch build; on Intel Macs it uses CPU.
CPU training can take a long time. The checkpoint stores class names and model settings
needed by the app. Training requires the dataset and internet access for the pretrained
ConvNeXt weights on the first run.

## 4. Launch the browser app

```bash
python -m satscene.app --config configs/default.yaml \
  --checkpoint runs/convnext_tiny/best_model.pth
```

This opens the modern browser interface at `http://127.0.0.1:7860`. Upload a satellite or
aerial image, then select **Analyze image** to see the top predictions, confidence scores,
and suggested follow-up. **Clear all** removes the uploaded image and resets the results.
To choose another checkpoint or port, pass a different `--checkpoint` path or `--port`.
The app binds to your own Mac by default; `--share` is optional and creates a public link.
Press Control-C in Terminal to stop the server.

### Optional native macOS desktop app

```bash
python -m satscene.desktop --config configs/default.yaml \
  --checkpoint runs/convnext_tiny/best_model.pth
```

The native Cocoa app opens a desktop window directly. Select or drop an image to start
analysis automatically. The PyObjC Cocoa bridge is required only for this optional app.

## 5. Evaluate the held-out test set

After model selection is complete, run this once:

```bash
python -m satscene.evaluate --config configs/default.yaml --split test
```

Metrics and a confusion matrix are saved under `runs/convnext_tiny/`. Keep test results
separate from decisions about model settings.

## Troubleshooting

- **Checkpoint not found:** train first, or pass an existing checkpoint with
  `--checkpoint`. The app does not silently create a fake prediction.
- **No labeled images found:** check the configured data root and confirm YOLO label
  text files exist beside the images under each split's `labels/` directory.
- **MPS unavailable:** SatScene falls back to CPU. Check `python -c 'import torch; print(torch.backends.mps.is_available())'`.
- **Port 7860 is occupied:** choose another port, for example `--port 7861`.
- **Kaggle download fails:** verify Kaggle credentials and that the dataset is accessible
  to your Kaggle account.

The current dataset consists of non-georeferenced JPEGs. The GUI classifies image
content; it does not read coordinates or execute NDVI, DEM, or other GIS analyses.
