# SatScene: industry-style experiment plan

## What the notebook currently demonstrates

The notebook documents a ConvNeXt-Tiny training attempt, but its executed code and
saved output disagree: the visible label code assigns every image to the first class,
while the output shows a 14-class distribution and high validation accuracy. Treat those
notebook metrics as historical, unverified output. Do not put them forward as a verified
benchmark. The packaged `src/satscene` workflow parses YOLO labels and is the source of
truth for a fresh run.

The current dataset has only 1,000 images, class labels are inferred from detection
annotations, and the JPEGs are not georeferenced. This supports a small remote-sensing
scene-classification project; it does not support claims of operational GIS analysis.

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

Run all model selection on train/validation only, with the test split untouched until the
final recipe is fixed:

1. Majority-class predictor (sanity baseline).
2. Frozen pretrained ConvNeXt backbone with a trained classification head.
3. Fine-tuned ConvNeXt-Tiny using the packaged training script.
4. Optional smaller deployment baseline, such as MobileNetV3, if time allows.

Use the same split, preprocessing, and evaluation code for each. Choose checkpoints by
validation macro-F1 (the current default) because the classes are imbalanced. Record
hardware, software versions, seed, config, duration, and checkpoint identifier.

### Gate 3 — Diagnose generalization

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

After choosing the method and settings, evaluate the test split once. Report the sample
count and uncertainty; do not use test outcomes to tune the recipe. For a stronger
industry-style demo, record median and 95th-percentile inference latency on the target Mac,
the selected backend (MPS or CPU), model size, and behavior when confidence is low. These
are deployment measurements, not model-quality substitutes. If you add an abstain/review
threshold, select it on validation data and report both coverage and error among accepted
predictions; do not tune that threshold on test data.

The current evaluator writes `metrics_test.json`, `predictions_test.csv`, and a combined
figure with count and normalized confusion matrices, per-class F1, and a reliability
diagram. Training writes per-epoch validation loss, accuracy, macro-F1, balanced accuracy,
and learning rate. Generate its plot with:

```bash
python scripts/plot_training_history.py runs/convnext_tiny/history.json
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

Avoid reporting only accuracy. The old 80.5% validation accuracy is checkpoint-selected,
and its originating notebook run is not verified because of the label-code/output conflict.

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

Replace every placeholder with outputs from the verified packaged run. If the test run or
error/duplicate audit is not complete, label that item as not completed instead of implying
it was done.
