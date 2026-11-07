# Seven-minute presentation content

Use this as the slide-by-slide content when preparing the required PowerPoint deck.
Report the archived held-out test evaluation accurately, without inferring model
provenance that its files do not contain. The assignment asks for
a PowerPoint presentation and a recording; this outline is not a substitute for either.

| Slide | Time | Content |
|---|---:|---|
| 1. SatScene and the task | 0:45 | Classify overhead images into 14 land-use classes. The dataset has no georeferencing, so this is image classification, not GIS raster analysis. |
| 2. Data and labels | 1:00 | 700/200/100 train/validation/test images. Labels come from the most frequent YOLO object class. Explain class imbalance and derived-label limitations. |
| 3. Method and software | 1:15 | Three-model comparison: ConvNeXtV2-Tiny, SwinV2-Tiny, and DINOv2 ViT-S/14 via `timm`; shared label parsing, training, validation, and evaluation. Explain what is third-party and what this project implements. |
| 4. Validation result | 1:00 | Archived comparison: ConvNeXtV2-Tiny 80.0% accuracy / 0.788 macro-F1 (epoch 9); SwinV2-Tiny 79.5% / 0.782 (epoch 10); DINOv2 38.0% / 0.305 (epoch 10). One seed per model; these are validation results, not test results. |
| 5. Held-out test result | 1:00 | The archived 100-image test artifact reports 86/100 correct, macro-F1 0.783, balanced accuracy 0.777, and top-3 accuracy 95%. Its files do not identify the checkpoint, so do not attribute it to the three comparison models. The supplied test split has already been inspected and must not be used for tuning. |
| 6. Software demo | 1:30 | Launch the local GUI, upload the Desert sample image, and show its matching prediction, alternatives, and follow-up suggestion. Treat one correct example as a software demonstration, not evidence of model reliability. |
| 7. Limits and next steps | 0:30 | Explain that the comparison results are validation-only and the archived test artifact has unknown model/checkpoint provenance. Do not tune on the already exposed test example; evaluate a selected comparison checkpoint on a fresh test set. |

Before recording, use the archived comparison metrics in `runs/README.md`. The test
example's model/checkpoint identity is not recorded. Replace any stale historical figures in the local `paper/main.pdf`,
confirm the final slide file uses the assignment's required filename, and add the video URL
to `surname_ha1_video.txt`.
