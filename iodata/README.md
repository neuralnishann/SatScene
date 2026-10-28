# Input and output data

This folder contains a small, traceable example of the software's input/output contract.
The complete 1,000-image dataset remains under `data/` (excluded from Git due to size).

- `sample_input/`: one held-out test image and its YOLO annotation.
- `sample_output/`: the prediction row for that image, plus the aggregate held-out metrics
  and confusion figure produced by the evaluation run.

Recreate the outputs after training with:

```bash
python -m satscene.evaluate --config configs/default.yaml --split test \
  --checkpoint runs/convnext_tiny/best_model.pth
```

The included historical output is a failure case: the model predicts Agriculture for all
100 test images (6% accuracy). Do not present it as a successful classifier result.
