# HA1 software-project submission checklist

## Included in this workspace

- [x] Project source under `src/satscene/` and tests under `tests/`.
- [x] Deployment instructions in `docs/Deployment_Guide.docx`.
- [x] Resource catalogue and source links in `resources/`.
- [x] Input/output example in `iodata/`.
- [x] Seven-slide PowerPoint deck: `Nishan_SMGazzaliArafat_individual_SatScene.pptx`.
- [x] AI declaration PDF draft: `Nishan_AI_in_my_work.pdf`.
- [x] Test-result artifacts are saved under `runs/convnext_tiny/` and copied into `iodata/sample_output/`.

## Author actions still required

- [ ] Edit the AI declaration to describe all earlier AI use (or explicitly say there was
  none) and replace the draft PDF with the completed declaration.
- [ ] Record the seven-minute presentation/demo and publish it to YouTube or Rutube; add
  the link to `surname_ha1_video.txt`.
- [ ] Open the generated PowerPoint in PowerPoint, review its rendering, personalize the
  filename for the actual group/topic naming rule, and replace the demo slide text with
  the chosen live-demo capture if needed.
- [ ] Rebuild `paper/main.pdf` from the updated `paper/main.tex`. The existing PDF is stale
  and still reports that the test split was not evaluated. The `paper/` directory is
  ignored by Git, so include the updated report source and PDF in the final archive.
- [ ] Investigate the validation/test collapse without tuning on the held-out test set.
  The current held-out result is 6% accuracy, with all predictions assigned to Agriculture.
- [ ] Run the unit-test suite in an environment with pytest installed and include the
  results if required by the instructor.
- [ ] Assemble the required single ZIP with root folder
  `Nishan_SMGazzaliArafat_HA1/`; include `resources/`, `iodata/`, source, report, deck,
  completed AI declaration, and video-link text file.
