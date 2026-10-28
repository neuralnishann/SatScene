# Seven-minute presentation content

Use this as the slide-by-slide content when preparing the required PowerPoint deck.
Keep the held-out test failure visible and explain it accurately. The assignment asks for
a PowerPoint presentation and a recording; this outline is not a substitute for either.

| Slide | Time | Content |
|---|---:|---|
| 1. SatScene and the task | 0:45 | Classify overhead images into 14 land-use classes. The dataset has no georeferencing, so this is image classification, not GIS raster analysis. |
| 2. Data and labels | 1:00 | 700/200/100 train/validation/test images. Labels come from the most frequent YOLO object class. Explain class imbalance and derived-label limitations. |
| 3. Method and software | 1:15 | ConvNeXt-Tiny via `timm`; custom parsing, training, validation, evaluation, rule-based suggestions, and local Gradio interface. Explain what is third-party and what this project implements. |
| 4. Validation result | 1:00 | Packaged seeded run: 77.5% validation accuracy, macro-F1 0.761, best epoch 4. One run only; validation was used to select the checkpoint. |
| 5. Held-out test result | 1:00 | Test: 6/100 correct, macro-F1 0.008. All predictions are Agriculture with confidence near 1.0. Treat as a failed result; cause is not yet known and test data must not be used for tuning. |
| 6. Software demo | 1:30 | Launch the local GUI, upload a sample image, show prediction, alternatives, and follow-up suggestion. Disclose that this checkpoint is unreliable; use the failure-case sample from `iodata/` or a screenshot. |
| 7. Limits and next steps | 0:30 | Diagnose the validation/test gap using training/validation evidence and data audits; improve the model only with train/validation; reserve a fresh final test for one-time reporting. |

Before recording, replace any stale historical figures in the local `paper/main.pdf`,
confirm the final slide file uses the assignment's required filename, and add the video URL
to `surname_ha1_video.txt`.
