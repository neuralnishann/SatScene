# SatScene: industry-style experiment plan

## What the notebook currently demonstrates

The current `GIS_project_original.ipynb` compares ConvNeXtV2-Tiny, SwinV2-Tiny, and
DINOv2 ViT-S/14. Its code calls `collect_split` from `src/satscene/labels.py` to derive
image labels from YOLO annotations. The saved output reports 700 labeled training images,
200 labeled validation images, all 14 classes, and no skipped images for those two splits.

The saved validation results are still limited evidence: each model has one recorded
seed, and checkpoints are selected using the same 200-image validation split used for
comparison. Report them as single-run validation results, not as a stable model ranking
or an estimate from the held-out test set. The packaged command-line trainer also uses
the shared `src/satscene` label parser and is a separate training workflow.

The current dataset has only 1,000 images, class labels are inferred from detection
annotations, and the JPEGs are not georeferenced. This supports a small remote-sensing
scene-classification project; it does not support claims of operational GIS analysis.

## Archived results at submission

The final saved comparison contains one run per model. All three runs use seed 42,
10 epochs, batch size 16, and validation macro-F1 checkpoint selection. Their validation
metrics are summarized below; detailed class reports, calibration data, predictions,
histories, and plots are in `runs/`.

| Model | Best epoch | Correct | Accuracy (95% Wilson CI) | Macro-F1 | Balanced accuracy | Top-3 |
|---|---:|---:|---:|---:|---:|---:|
| ConvNeXtV2-Tiny | 9 | 160/200 | 80.0% (73.9–85.0%) | 0.7881 | 0.7876 | 94.0% |
| SwinV2-Tiny | 10 | 159/200 | 79.5% (73.4–84.5%) | 0.7817 | 0.7770 | 93.5% |
| DINOv2 ViT-S/14 | 10 | 76/200 | 38.0% (31.6–44.9%) | 0.3049 | 0.3441 | 69.0% |

These are validation results from a single seed per model, not a stable ranking or
held-out performance estimate. The best ConvNeXtV2-Tiny result exceeds SwinV2-Tiny by
one validation image; this small difference does not establish a dependable advantage.

A separate historical test artifact in `iodata/sample_output/metrics_test.json` records
86/100 correct (86.0% accuracy), macro-F1 0.7829, balanced accuracy 0.7772, and
top-3 accuracy 95.0%. The included files do not identify the model or checkpoint, so
these metrics cannot be attributed to any of the three comparison checkpoints. Because
this test example is already available and inspected, it cannot serve as a blind final
assessment. Comparison checkpoints are available in the local
`runs/<model>/best_model.pth` folders but are ignored by Git because of their size.

## Recommended project sequence

### Gate 1 — Lock down the data

1. Record dataset version, license, split sizes, class names, and the label derivation rule.
2. Count missing/empty annotation files and invalid class IDs. Save the class distribution.
3. Visually inspect a stratified sample of images with their YOLO boxes and derived labels.
4. Audit exact and near-duplicate images across train, validation, and test; report the
   method and findings. Never move samples between the supplied test split and training.
5. State that the most-frequent object class is only a proxy scene label; discuss mixed scenes.

**Deliverable:** a data-quality table and one annotated image montage. Stop if the label
mapping or split counts cannot be reconciled.

### Gate 2 — Establish honest baselines

Model selection for a clean experiment must use train/validation only and leave test
untouched until the recipe is fixed. In this repository, the three archived comparison
runs followed that rule; a separate packaged-workflow test result has since been exposed,
so this supplied test split is no longer blind for final evaluation:

1. Majority-class predictor (sanity baseline).
2. Frozen pretrained ConvNeXt backbone with a trained classification head.
3. Fine-tuned ConvNeXt-Tiny using the packaged training script.
4. Optional smaller deployment baseline, such as MobileNetV3, if time allows.

Use the same split, preprocessing, and evaluation code for each. The saved comparison
selects checkpoints by validation macro-F1; the packaged trainer defaults to the same
metric. Record hardware, software versions, seed, config, duration, and checkpoint
identifier.

### Gate 3 — Diagnose generalization

The initial ConvNeXtV2 fine-tuning candidate is available in
`configs/convnextv2_finetune.yaml`. It uses a lower backbone learning rate than head
learning rate, one warm-up epoch, cosine decay, and validation-accuracy checkpoint
selection with macro-F1 and balanced accuracy still recorded. Its output is isolated in
`runs/convnextv2_lr_tune/`; it has not been run yet and is not a measured improvement.

1. Train at least three seeds for the selected recipe if compute permits; report mean and
   standard deviation for validation metrics. Keep single-run conclusions clearly marked.
2. Compare one controlled ablation at a time: class-weighted loss, stronger augmentation,
   label smoothing, and/or partial layer freezing. Do not claim a gain unless validation
   results support it.
3. Review false positives and false negatives. Group recurring errors by visually similar
   classes and annotation ambiguity; include representative examples.
4. Consider confidence calibration. ECE and Brier score are now emitted by test evaluation;
   low calibration quality means displayed softmax confidence should not be read as a
   probability of correctness.
5. Optional explainability extension: add Grad-CAM overlays for correctly and incorrectly
   classified samples, and describe them as model-attention diagnostics rather than proof
   of causal reasoning.

### Gate 4 — Final held-out test and deployment evidence

The packaged-workflow test example has already been evaluated and exposed in this
repository, so do not treat the supplied test split as a blind final assessment or tune on
that result. A future final evaluation requires a fresh, untouched test set. Report the
sample count and uncertainty for any such evaluation. For a stronger
industry-style demo, record median and 95th-percentile inference latency on the target Mac,
the selected backend (MPS or CPU), model size, and behavior when confidence is low. These
are deployment measurements, not model-quality substitutes. If you add an abstain/review
threshold, select it on validation data and report both coverage and error among accepted
predictions; do not tune that threshold on test data.

The evaluator writes `metrics_test.json`, `predictions_test.csv`, and a combined
figure with count and normalized confusion matrices, per-class F1, and a reliability
diagram. Training writes per-epoch validation loss, accuracy, macro-F1, balanced accuracy,
and learning rate. Generate its plot with:

```bash
python scripts/plot_training_history.py runs/convnextv2_tiny/history.json
```

## Metrics to show

| Metric | Why show it | Caveat |
|---|---|---|
| Accuracy + 95% Wilson interval | Easy overall headline and uncertainty | Can hide poor minority-class performance |
| Macro-F1 | Gives each class equal influence | Can be unstable for classes with very few examples |
| Balanced accuracy | Mean recall across classes | Does not describe precision or probability quality |
| Per-class precision, recall, F1, support | Shows which classes work and how much evidence each has | Always show support alongside scores |
| Confusion matrix, counts and row-normalized | Makes class confusions interpretable | Counts are essential when support differs |
| Top-3 accuracy | Useful when the system is a human-review aid | Does not replace top-1 or macro-F1 |
| Cross-entropy loss | Captures confidence-sensitive prediction quality | Compare only under the same label set and setup |
| ECE and multiclass Brier score | Diagnose confidence calibration | ECE depends on binning; state its bin count |
| Train/validation loss and metric curves | Reveals overfitting and training instability | Training accuracy is augmented/train-mode in this implementation |

Avoid reporting only accuracy. The archived ConvNeXtV2-Tiny result is 80.0% validation
accuracy on 200 images (160 correct), selected from one run. Its 95% Wilson interval is
approximately 73.9–85.0%; this interval does not capture variation from retraining.

## Presentation flow (7 minutes)

1. **Problem and scope (45 s):** overhead scene classification; explain the non-georeferenced
   dataset and the limits of the agent's rule suggestions.
2. **Data and label quality (60 s):** class counts, derived-label example, imbalance, split
   and duplicate audit status.
3. **Method (75 s):** majority and pretrained baselines, ConvNeXt fine-tuning, augmentation,
   model-selection rule, and reproducibility settings.
4. **Learning behavior (60 s):** train/validation loss and macro-F1 curves; explain overfit
   evidence and what changed after ablations.
5. **Held-out results (90 s):** accuracy interval, macro-F1, per-class table, normalized
   confusion matrix, calibration, and limitations from sample size.
6. **Live GUI (60 s):** upload a sample, show top-3 and follow-up suggestion, demonstrate
   low-confidence behavior; state that GIS analyses are not executed.
7. **Conclusion (30 s):** one defensible claim, remaining risks, and next production step.

Use the archived comparison values above when describing validation performance. Do not
attribute the historical test artifact to the comparison models without model/checkpoint
provenance. The near-duplicate audit and a held-out evaluation of the selected comparison
checkpoint are not included in this repository snapshot; describe them as incomplete.
The supplied test example has already been inspected, so only a fresh test set could provide blind final
evidence.
