# Sample input and output

This folder demonstrates SatScene's image and evaluation-output formats. The input
image is from the supplied 100-image test split; its YOLO annotation assigns it the
label Desert.

## Included files

- `sample_input/`: one test image and its YOLO annotation.
- `sample_output/prediction.csv`: the image's prediction row. It records Desert as
  the predicted class, matching the true label, at 99.61% confidence.
- `sample_output/metrics_test.json`: aggregate metrics for a 100-image test evaluation:
  86.0% accuracy (86/100; 95% Wilson CI 77.9–91.5%), macro-F1 0.783, balanced
  accuracy 0.777, and top-3 accuracy 95.0%.
- `sample_output/confusion_test.png`: confusion matrices, per-class F1 scores, and a
  reliability diagram for that evaluation.

The evaluation files do not record the model or checkpoint identity. Do not attribute
these test metrics to one of the three archived comparison checkpoints without
independent provenance. The comparison results in [`../runs/README.md`](../runs/README.md)
are validation-only. The supplied test split has already been evaluated and is no
longer a blind final assessment.

The complete image dataset is not included; it remains under `data/` when downloaded
locally. To generate evaluation outputs for a checkpoint, run:

```bash
python -m satscene.evaluate --config configs/default.yaml --split test \
  --checkpoint /path/to/best_model.pth
```

Evaluation writes metrics, predictions, and diagnostic outputs to the configured
run directory. This command does not reproduce the archived files unless the same
checkpoint and evaluation setup are supplied.
