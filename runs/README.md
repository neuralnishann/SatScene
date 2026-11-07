# Archived training and evaluation runs

This folder contains the saved results for the three-model comparison in
`notebooks/GIS_project_original.ipynb`. These are one-run-per-model validation
artifacts, not repeated-seed estimates and not held-out test results. Each run used
seed 42, 10 epochs, batch size 16, AdamW (learning rate `3e-4`, weight decay `1e-4`),
train-only horizontal/vertical flips and 90-degree rotations (each p=0.5), and
checkpoint selection by validation macro-F1. The supplied 700/200 train/validation
splits were used; the test split was not part of this comparison.

| Folder | Model | Input | Best epoch | Validation accuracy | Macro-F1 | Balanced accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `convnextv2_tiny/` | ConvNeXtV2-Tiny (`convnextv2_tiny.fcmae_ft_in1k`) | 224 | 9 | 80.0% (160/200) | 0.7881 | 0.7876 |
| `swinv2_tiny_window8/` | SwinV2-Tiny (`swinv2_tiny_window8_256.ms_in1k`) | 256 | 10 | 79.5% (159/200) | 0.7817 | 0.7770 |
| `dinov2_vit_small_patch14/` | DINOv2 ViT-S/14 (`vit_small_patch14_dinov2.lvd142m`) | 224 | 10 | 38.0% (76/200) | 0.3049 | 0.3441 |

## Files

Each model directory contains its run config, epoch history, validation metrics,
per-image predictions, per-class metrics, confusion matrices, a diagnostic plot, and
the locally available `best_model.pth` checkpoint. The checkpoint files were found
and their embedded model names, image sizes, and class-name lists match their run
configs. They are about 83–107 MiB each and are ignored by Git, so a clone or archive
that excludes ignored files will need the weights copied in separately. Paths to the
original local dataset are intentionally omitted from the portable run configs.

Use these local checkpoint paths with `satscene.app`, `satscene.desktop`, or
`satscene.evaluate`:

```text
runs/convnextv2_tiny/best_model.pth
runs/swinv2_tiny_window8/best_model.pth
runs/dinov2_vit_small_patch14/best_model.pth
```

`model_comparison_metrics.csv` is the compact tabular source for the README result
table. JSON metric files contain the detailed class and calibration breakdowns.

## Other run routes

- `configs/default.yaml` describes a separate packaged ConvNeXt-Tiny training
  pipeline. Its output directory is `runs/convnext_tiny/`, which is not present in the
  current workspace. Its default app command therefore fails until that model is
  trained. The historical test example is under `iodata/sample_output/`; its files do
  not identify the model/checkpoint, so do not attribute it to a comparison run.
- `configs/convnextv2_finetune.yaml` describes a candidate experiment writing to
  `runs/convnextv2_lr_tune/`. No run artifacts for that candidate are present, so it
  has no reported result.
- `results/training_log.json` is an older, manually transcribed and unseeded notebook
  history. See `results/README.md`; it is distinct from the three saved runs above.
