# Seven-minute software demo recording checklist

The assignment asks for a recorded, seven-minute PowerPoint demo and a separate text file
containing its YouTube/Rutube link. No recording or public video link is included in this
workspace; record and publish it before submission, using a private link if desired.

Suggested timing (7:00 total):

1. Problem and scope — 0:45
2. Dataset and label derivation — 1:00
3. Three-model comparison and software workflow — 1:15
4. Test result and failure analysis — 1:30
5. Live GUI upload and result — 1:00
6. Limitations and next steps — 1:00
7. Closing — 0:30

Save the final URL in `surname_ha1_video.txt`. Do not overstate the classifier's reliability: the archived comparison is validation-only:
ConvNeXtV2-Tiny scored 80.0% (macro-F1 0.788), SwinV2-Tiny 79.5% (0.782), and DINOv2
38.0% (0.305), each on 200 images and one seed. Separately, the historical 100-image
test artifact reports 86% accuracy and macro-F1 0.783, but its files do not record the
model/checkpoint identity, so it is not evidence about those comparison checkpoints.
Show this distinction clearly. The supplied test example has already been inspected and
is not a blind final test set. Do not use it for model selection or claim a comparison
checkpoint was tested.
