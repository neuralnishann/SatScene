# Project resources

This folder catalogs the external dataset, research papers, and software used by
SatScene. The project bibliography is in the [main README](../README.md#references-and-licenses);
the exact Python dependency specifications are maintained in
[`requirements.txt`](../requirements.txt) and [`pyproject.toml`](../pyproject.toml).

## Dataset

- Umer Adnaan, *Remote Sensing Satellite Images* (Kaggle):
  <https://www.kaggle.com/datasets/umeradnaan/remote-sensing-satellite-images>
- The project uses the Roboflow Universe YOLOv8 version 2 export:
  <https://universe.roboflow.com/image-captioning-for-remote-sensingdata-0zids/remote-sensing-data/dataset/2>
- [`dataset_source.txt`](dataset_source.txt) records the source, export version, and
  attribution. The downloaded export reports CC BY 4.0; preserve attribution and verify
  the terms at the source before redistribution. The Kaggle listing and Roboflow export
  may specify different licenses, so use the terms for the exact copy being redistributed.
- The image dataset is kept separately under `data/` and is not copied into this folder.
  The export contains 1,000 images; see the project README for the split and label details.

## Models and research papers

The package workflow uses ConvNeXt-Tiny; the comparison notebook evaluates ConvNeXtV2-Tiny,
SwinV2-Tiny, and DINOv2 ViT-S/14. Their implementations and pretrained weights are loaded
through [`timm`](https://github.com/huggingface/pytorch-image-models).

The notebook fine-tunes the backbones on the supplied train split, selects checkpoints
using validation macro-F1, and evaluates on the held-out test split. Each model has one
recorded run, so the results do not establish a stable ranking.

## Complete project bibliography

These are the references listed in the main project README, including the model papers,
optimization and evaluation methods, remote-sensing background and benchmarks, software
documentation, and dataset source.

1. Z. Liu et al., “A ConvNet for the 2020s,” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2022. [CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html).
2. S. Woo et al., “ConvNeXt V2: Co-designing and scaling ConvNets with masked autoencoders,” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2023. [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Woo_ConvNeXt_V2_Co-Designing_and_Scaling_ConvNets_With_Masked_Autoencoders_CVPR_2023_paper.html).
3. Z. Liu et al., “Swin Transformer: Hierarchical vision Transformer using shifted windows,” in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, 2021. [CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.html).
4. Z. Liu et al., “Swin Transformer V2: Scaling up capacity and resolution,” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2022. [CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Swin_Transformer_V2_Scaling_Up_Capacity_and_Resolution_CVPR_2022_paper.html).
5. A. Dosovitskiy et al., “An image is worth 16×16 words: Transformers for image recognition at scale,” in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2021. [OpenReview](https://openreview.net/forum?id=YicbFdNTTy).
6. M. Oquab et al., “DINOv2: Learning robust visual features without supervision,” arXiv:2304.07193, 2023. [arXiv](https://arxiv.org/abs/2304.07193).
7. I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2019. [OpenReview](https://openreview.net/forum?id=Bkg6RiCqY7).
8. I. Loshchilov and F. Hutter, “SGDR: Stochastic gradient descent with warm restarts,” in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2017. [arXiv](https://arxiv.org/abs/1608.03983).
9. G. W. Brier, “Verification of forecasts expressed in terms of probability,” *Mon. Weather Rev.*, vol. 78, no. 1, pp. 1–3, Jan. 1950. doi: `10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2`.
10. C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in *Proc. 34th Int. Conf. Mach. Learn. (ICML)*, PMLR, vol. 70, 2017, pp. 1321–1330. [PMLR](https://proceedings.mlr.press/v70/guo17a.html).
11. E. B. Wilson, “Probable inference, the law of succession, and statistical inference,” *J. Amer. Stat. Assoc.*, vol. 22, no. 158, pp. 209–212, Jun. 1927. doi: `10.1080/01621459.1927.10502953`.
12. R. G. Newcombe, “Two-sided confidence intervals for the single proportion: Comparison of seven methods,” *Stat. Med.*, vol. 17, no. 8, pp. 857–872, Apr. 1998. doi: `10.1002/(SICI)1097-0258(19980430)17:8<857::AID-SIM777>3.0.CO;2-E`.
13. S. Dutta and M. Das, “Remote sensing scene classification under scarcity of labelled samples—A survey of the state-of-the-arts,” *Comput. Geosci.*, vol. 171, Art. no. 105295, 2023. [doi:10.1016/j.cageo.2022.105295](https://doi.org/10.1016/j.cageo.2022.105295).
14. G.-S. Xia et al., “AID: A benchmark data set for performance evaluation of aerial scene classification,” *IEEE Trans. Geosci. Remote Sens.*, vol. 55, no. 7, pp. 3965–3981, Jul. 2017. doi: `10.1109/TGRS.2017.2685945`.
15. G. Cheng, J. Han, and X. Lu, “Remote sensing image scene classification: Benchmark and state of the art,” *Proc. IEEE*, vol. 105, no. 10, pp. 1865–1883, Oct. 2017. [arXiv](https://arxiv.org/abs/1703.00121).
16. P. Helber et al., “EuroSAT: A novel dataset and deep learning benchmark for land use and land cover classification,” *IEEE J. Sel. Topics Appl. Earth Observ. Remote Sens.*, vol. 12, 2019. [arXiv](https://arxiv.org/abs/1709.00029).
17. G. Sumbul, M. Charfuelan, B. Demir, and V. Markl, “BigEarthNet: A large-scale benchmark archive for remote sensing image understanding,” in *Proc. IEEE Int. Geosci. Remote Sens. Symp. (IGARSS)*, 2019, pp. 5901–5904. [arXiv](https://arxiv.org/abs/1902.06148).
18. PyTorch, “CrossEntropyLoss,” “AdamW,” and “CosineAnnealingLR,” *PyTorch Documentation*. [PyTorch](https://docs.pytorch.org/docs/stable/).
19. R. Wightman et al., “PyTorch Image Models (timm),” *timm Documentation*. [timm](https://huggingface.co/docs/timm/reference/models).
20. “Remote Sensing Satellite Images,” Kaggle dataset; Roboflow, “Remote Sensing Data,” YOLOv8 export. [Kaggle](https://www.kaggle.com/datasets/umeradnaan/remote-sensing-satellite-images) and [Roboflow Universe](https://universe.roboflow.com/image-captioning-for-remote-sensingdata-0zids/remote-sensing-data/dataset/2).
21. G. V. Lakshmi, P. Prathik, and G. Geetha, “Satellite Image Classification Using Deep Learning,” in *Proc. 2024 2nd International Conference on Signal Processing, Communication, Power and Embedded System (SCOPES)*, Paralakhemundi Campus, Centurion University of Technology and Management, Odisha, India, 2024, pp. 1–5. [doi:10.1109/SCOPES64467.2024.10990975](https://doi.org/10.1109/SCOPES64467.2024.10990975).
22. D. Yadav, K. Kapoor, A. K. Yadav, M. Kumar, A. Jain, and J. Morato Lara, “Satellite image classification using deep learning approach,” *Earth Sci. Inform.*, vol. 17, no. 3, pp. 2495–2508, 2024. [doi:10.1007/s12145-024-01301-x](https://doi.org/10.1007/s12145-024-01301-x).

## Software resources

The project requires Python 3.10 or later. Install the declared dependencies from the
repository root with `pip install -r requirements.txt`.

| Resource | Use in the project |
| --- | --- |
| [PyTorch](https://pytorch.org/) | Model training and inference |
| [timm](https://huggingface.co/docs/timm) | Model architectures and pretrained weights |
| [Albumentations](https://albumentations.ai/) | Image transformations |
| [OpenCV](https://opencv.org/) and [Pillow](https://python-pillow.org/) | Image loading and processing |
| [scikit-learn](https://scikit-learn.org/) | Evaluation metrics and data utilities |
| [NumPy](https://numpy.org/), [Matplotlib](https://matplotlib.org/), and [seaborn](https://seaborn.pydata.org/) | Numerical work and result visualization |
| [PyYAML](https://pyyaml.org/) and [tqdm](https://tqdm.github.io/) | Configuration and progress reporting |
| [Gradio](https://www.gradio.app/) | Local prediction interface |
| [Kaggle API](https://github.com/Kaggle/kaggle-api) | Dataset download support |

The optional `dev` extra in `pyproject.toml` provides pytest; the `presentation` extra
provides python-pptx. macOS additionally installs the Cocoa bridge specified in the
dependency manifests.

## Project metadata

- [`CITATION.cff`](../CITATION.cff): software citation metadata.
- [`LICENSE`](../LICENSE): license for this project's code; third-party data and model
  resources retain their own licenses.
