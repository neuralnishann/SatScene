# Project resources

This folder lists the external data, model, and implementation resources used by SatScene.
The files in this folder are included in the final project archive; the dataset itself is
kept in `data/` because it is a separate 55 MB download.

## Dataset

- Umer Adnaan, *Remote Sensing Satellite Images* (Kaggle):
  <https://www.kaggle.com/datasets/umeradnaan/remote-sensing-satellite-images>
- Roboflow dataset version and license information: see `dataset_source.txt` and the
  upstream `README.roboflow.txt`.
- The local export reports CC BY 4.0. Preserve attribution and check the upstream terms
  before redistribution. The Kaggle listing and Roboflow export may describe different
  licensing terms; this project follows the license in the exact downloaded export.

## Selected method and model implementation

- ConvNeXt paper: Liu et al., *A ConvNet for the 2020s*, CVPR 2022,
  <https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html>
- ConvNeXt implementation and pretrained weights: `timm`,
  <https://github.com/huggingface/pytorch-image-models>
- Albumentations image transforms: <https://albumentations.ai/>
- PyTorch: <https://pytorch.org/>
- Gradio local interface: <https://www.gradio.app/>

The reference paper is listed here as the primary method reference. SatScene implements
the data parsing, training, validation, evaluation, reporting, and interface around the
ConvNeXt model. The ConvNeXt architecture and pretrained weights come from `timm`.

## Three comparison architectures in the Colab notebook

- ConvNeXt V2: Woo et al., *ConvNeXt V2: Co-designing and Scaling ConvNets with Masked
  Autoencoders*, <https://arxiv.org/abs/2301.00808>.
- Swin Transformer V2: Liu et al., *Swin Transformer V2: Scaling Up Capacity and
  Resolution*, <https://arxiv.org/abs/2111.09883>.
- DINOv2 ViT-S/14: Oquab et al., *DINOv2: Learning Robust Visual Features without
  Supervision*, <https://arxiv.org/abs/2304.07193>.

The comparison notebook uses pretrained ImageNet/self-supervised weights from `timm` and
fine-tunes the classifier head and backbone on the dataset. It saves each selected
checkpoint by validation macro-F1. Dataset size is small, so results must be compared on
the same supplied train/validation split and the held-out test set must stay out of tuning.

## Local resources

- `dataset_source.txt`: attribution and license note copied from the downloaded export.
- `../docs/EXPERIMENT_PLAN.md`: evaluation and analysis protocol.
- `../docs/Deployment_Guide.docx`: installation and deployment guide.
- `../CITATION.cff` and `../LICENSE`: citation metadata and project license.
