# Historical notebook results

`training_log.json` is a manual transcription of the saved cell outputs from
`notebooks/GIS_project_original.ipynb` (Colab/GPU). It records an older
`convnext_tiny.fb_in22k_ft_in1k` run with batch size 32 and 224-pixel input. The
run was unseeded, and its training metrics were measured with augmentation enabled.
It is retained as provenance, not as the final three-model comparison or as a
reproducible run record.

- `training_log.json`: epoch metrics and the recorded train-class distribution.
- `figures/training_curves.png`: plot generated from that history.
- `figures/class_distribution.png`: class-count visualization associated with the
  notebook data inspection.

The results from the later, saved three-model comparison are in `../runs/` and are
summarized in [`../runs/README.md`](../runs/README.md). Do not combine metrics from
these separate runs into a single ranking.
