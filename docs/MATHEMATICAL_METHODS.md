# Mathematical foundations of SatScene

This note defines the mathematical objects used by the code and README. The
formulations describe the implemented 14-class image classifier and its metrics;
they do not imply that the archived single-run measurements are statistically
stable. Model blocks are instantiated by `timm` [19], while this repository supplies
the task head, data path, optimization loop, checkpointing, and evaluation.
Equations use GitHub-supported LaTeX delimiters so they render as mathematical
notation in the repository's Markdown view.

## 1. Task, observations, and target construction

Let an RGB image be $x_i \in [0,255]^{H\times W\times 3}$. The dataset contains $K=14$
scene labels. YOLO files describe detected objects rather than image-level
scenes. For annotation class IDs $a_{i1},\ldots,a_{iJ}$, define the class counts
$n_{ic}=\sum_{j=1}^{J}\mathbf{1}[a_{ij}=c]$. SatScene constructs the target by
finding the first annotation whose class has maximal frequency:

$$
j_i^*=\min\left\{j:n_{i,a_{ij}}=\max_{c}n_{ic}\right\},\qquad
y_i=a_{i,j_i^*}\in\{0,\ldots,K-1\}.
$$

An image with a missing or empty annotation file is excluded. This deterministic
mapping converts an object-label collection to one scene class, but it is a
proxy target: an image may contain multiple scene types or a dominant object
that does not describe its overall land use. Label noise therefore enters the
supervised objective before model error is considered.

For resized channel $q\in\{R,G,B\}$, preprocessing is

$$
x'_{i,h,w,q}=\frac{x_{i,h,w,q}/255-\mu_q}{\sigma_q},\qquad
\boldsymbol{\mu}=(0.485,0.456,0.406),\quad
\boldsymbol{\sigma}=(0.229,0.224,0.225).
$$

These are ImageNet normalization constants. Training samples horizontal flip,
vertical flip, and a multiple-of-90-degree rotation independently with
probability 0.5 each; validation and evaluation use no random augmentation.
The transforms are intended to preserve label under overhead orientation, an
assumption that should be checked against the image source and task.

## 2. Shared classifier formulation

For each candidate backbone $m$, the feature extractor is
$\mathbf{h}_i=f_{\theta_m}(\mathbf{x}'_i)\in\mathbb{R}^{d}$. A task-specific affine head produces $K$ logits:

$$
\mathbf{z}_i=\mathbf{W}_m\mathbf{h}_i+\mathbf{b}_m,\qquad
\mathbf{W}_m\in\mathbb{R}^{K\times d},\quad \mathbf{b}_m\in\mathbb{R}^{K}.
$$

$$
p_{ic}=\operatorname{softmax}(\mathbf{z}_i)_c=
\frac{\exp(z_{ic})}{\sum_{j=0}^{K-1}\exp(z_{ij})},\qquad
\hat{y}_i=\arg\max_{c\in\{0,\ldots,K-1\}}p_{ic}.
$$

The chosen `timm` classifier implementation may include architecture-specific
pooling and normalization before the final affine layer. The new output dimension
is 14. The comparison notebook fine-tunes the pretrained backbone and head; it
does not freeze the backbone. The per-example probabilities sum to one, but are
not necessarily calibrated estimates of correctness.

### 2.1 ConvNeXtV2-Tiny

ConvNeXtV2 [1,2] is a hierarchical convolutional feature extractor. For the recorded
224 × 224 input, the stem and successive reductions create feature maps at
decreasing spatial resolutions; the Tiny stage depths are `(3, 3, 9, 3)`.
Depthwise spatial convolutions extract local patterns, while pointwise channel
mixing and nonlinearities build more complex features. This gives the model a
locality bias and progressively wider receptive fields, which can be useful for
land-cover texture and spatial arrangement.

The V2 block adds Global Response Normalization (GRN) [2]. For feature tensor
$\mathbf{X}\in\mathbb{R}^{H\times W\times C}$, one channel-wise form is

$$
G_c=\sqrt{\sum_{h=1}^{H}\sum_{w=1}^{W}X_{h,w,c}^{2}},\qquad
n_c=\frac{G_c}{\frac{1}{C}\sum_{j=1}^{C}G_j+\varepsilon},
$$

$$
\operatorname{GRN}(\mathbf{X})_{h,w,c}
=X_{h,w,c}+\gamma_c\left(X_{h,w,c}n_c\right)+\beta_c.
$$

$\gamma_c$ and $\beta_c$ are learned channel parameters and $\varepsilon$ stabilizes division. GRN
contrasts global channel responses; it is not a normalization of the input RGB
image. ConvNeXtV2 was co-designed with fully convolutional masked autoencoding
pretraining, while this experiment loads an ImageNet fine-tuned pretrained tag.
The reported results therefore evaluate the selected checkpoint and downstream
fine-tuning recipe, not masked pretraining performed by this project.

### 2.2 Swin Transformer V2-Tiny

SwinV2 begins with patch embedding and applies self-attention in local windows [3,4].
Alternating regular and shifted windows allow information to cross window
boundaries; patch merging creates a hierarchy of coarser features. In a window
with $M\times M$ tokens, attention among its tokens has an interaction term
proportional to $M^4d$. If there are $N$ total tokens, window attention is approximately
$\mathcal{O}(NM^2d)$, compared with $\mathcal{O}(N^2d)$ for global attention at that resolution.
The actual implementation also includes projections, MLPs, and multiple stages.

SwinV2 uses scaled cosine attention [4]. For query token $i$, key token $j$,
and attention head $h$, a simplified score is

$$
A_{ij}^{(h)}=\tau_h\left(\frac{\mathbf{q}_i^{(h)}}{\lVert\mathbf{q}_i^{(h)}\rVert_2}\right)^{\!\top}
\left(\frac{\mathbf{k}_j^{(h)}}{\lVert\mathbf{k}_j^{(h)}\rVert_2}\right)
+B_{ij}^{(h)},\qquad
\mathbf{o}_i^{(h)}=\sum_j\operatorname{softmax}_j(A_{ij}^{(h)})\mathbf{v}_j^{(h)}.
$$

Here $\tau_h$ is a learned scale and $B_{ij}^{(h)}$ is a relative-position bias. This
description omits masking and the architecture's normalization/residual details.
The comparison uses the 256 × 256, window-8 pretrained configuration. Its
hierarchical local context is a plausible fit for scenes containing structures
at different spatial scales; whether it helps must be established empirically.

For a hand-checkable attention-size comparison, the first Swin stage takes the
256-pixel input to $64\times64=4{,}096$ patch tokens. An $8\times8$ window partition has
$4{,}096/64=64$ windows, each with $64^2=4{,}096$ query-key score entries, or
$64\times4{,}096=262{,}144$ entries per head for the stage. Global attention over the
same $4{,}096$ tokens would form $4{,}096^2=16{,}777{,}216$ entries per head. This is a
64-fold reduction in pairwise score entries at that stage, not a 64-fold claim
about whole-model latency or total FLOPs. For DINOv2's sequence of 257 tokens,
global attention forms $257^2=66{,}049$ entries per head per block.

### 2.3 DINOv2 ViT-S/14

For a $224\times224$ image and patch size $P=14$, the number of image patch tokens
is $(224/14)^2=16^2=256$. Adding a class token gives a sequence length of
$N=257$ (the run's model configuration does not use register tokens). A patch
embedding maps each flattened $14\times14\times3$ patch into a token vector. Positional
information is added before Transformer blocks [5,6]. For one self-attention head,

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V,\qquad
\operatorname{Attn}(X)=\operatorname{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V.
$$

Global attention couples all `N` tokens, with pairwise interaction cost
$\mathcal{O}(N^2d)$ per block (more fully, attention products scale as $\mathcal{O}(N^2d)$ and
projections/MLPs add terms proportional to $Nd^2$). The final class-token
representation feeds the 14-class head. The pretrained representation is from
DINOv2 self-supervised training; SatScene fine-tunes it with scene labels and
does not reproduce DINOv2's teacher/student, iBOT, or masked patch pretraining.

## 3. Objective, optimizer, and checkpoint selection

For a batch of size $B$, the unweighted cross-entropy objective implemented in
the model comparison [18] is

$$
\mathcal{L}_{\mathrm{CE}}
=-\frac{1}{B}\sum_{i=1}^{B}\log p_{i,y_i}
=\frac{1}{B}\sum_{i=1}^{B}\left[-z_{i,y_i}
+\log\!\left(\sum_{c=0}^{K-1}e^{z_{i,c}}\right)\right].
$$

The second expression is the numerically stable log-sum-exp form. The packaged
trainer can optionally use inverse-frequency class weights
$w_c=N_{\mathrm{train}}/(K n_c)$ and weighted mean loss
$\sum_i w_{y_i}\ell_i/\sum_i w_{y_i}$; that option is disabled in
`configs/default.yaml` and is not used in the three-model comparison.

The comparison uses AdamW [7] ($\eta=3\times10^{-4}$, weight decay $\lambda=10^{-4}$,
batch size 16, 10 epochs). For gradient $g_t$, first and second moments follow
$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t$ and
$v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$. Bias-corrected moments are
$\hat{m}_t=m_t/(1-\beta_1^t)$ and $\hat{v}_t=v_t/(1-\beta_2^t)$. AdamW applies
decoupled decay separately from the loss-gradient update:

$$
\theta_{t+1}=\theta_t-\eta_t\left(
\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\varepsilon}+\lambda\theta_t\right).
$$

The comparison code overrides only learning rate and weight decay; PyTorch's
AdamW defaults are $\beta_1=0.9$, $\beta_2=0.999$, and $\varepsilon=10^{-8}$.

The learning rate is cosine-annealed by epoch [8], with $T_{\max}=10$ and default
$\eta_{\min}=0$:

$$
\eta_t=\eta_{\min}+\frac{1}{2}(\eta_{\max}-\eta_{\min})
\left[1+\cos\!\left(\frac{\pi t}{T_{\max}}\right)\right].
$$

At each epoch, validation macro-F1 is computed and the highest-scoring epoch is
saved. This makes validation part of model selection; the best validation score
is optimistically biased as an estimate of future performance. The test split
must be used only after the model and settings are fixed.

The notebook comparison and packaged trainer are distinct runs. The comparison
uses batch size 16, model-specific input sizes, and the above 10-epoch protocol.
The default packaged config uses the `convnext_tiny.fb_in22k_ft_in1k` model,
224-pixel input, and batch size 32. Do not transfer the comparison score to the
packaged checkpoint or describe the packaged run as the validation winner.

## 4. Evaluation metrics in this repository

Let $C_{ab}$ be the number of samples with true class $a$ and predicted class
$b$. For class $c$,

$$
\begin{aligned}
\mathrm{TP}_c&=C_{cc}, &
\mathrm{FN}_c&=\sum_{b\ne c}C_{cb}, &
\mathrm{FP}_c&=\sum_{a\ne c}C_{ac},\\
\mathrm{Precision}_c&=\frac{\mathrm{TP}_c}{\mathrm{TP}_c+\mathrm{FP}_c}, &
\mathrm{Recall}_c&=\frac{\mathrm{TP}_c}{\mathrm{TP}_c+\mathrm{FN}_c}, &
\mathrm{F1}_c&=\frac{2\,\mathrm{Precision}_c\,\mathrm{Recall}_c}
{\mathrm{Precision}_c+\mathrm{Recall}_c}.
\end{aligned}
$$

Undefined zero-denominator class scores are set to zero by the implementation.
The reported aggregate scores are

$$
\begin{aligned}
\mathrm{Accuracy}&=\frac{\sum_{c=0}^{K-1}C_{cc}}{N}, &
\mathrm{MacroF1}&=\frac{1}{K}\sum_{c=0}^{K-1}\mathrm{F1}_c,\\
\mathrm{BalancedAcc}&=\frac{1}{K}\sum_{c=0}^{K-1}\mathrm{Recall}_c, &
\mathrm{WeightedF1}&=\sum_{c=0}^{K-1}\frac{n_c}{N}\mathrm{F1}_c,\\
\mathrm{TopKAcc}&=\frac{1}{N}\sum_{i=1}^{N}
\mathbf{1}\!\left[y_i\in\operatorname{TopK}(\mathbf{p}_i)\right].
\end{aligned}
$$

In single-label multiclass classification, macro recall equals balanced
accuracy, but macro-F1 and weighted-F1 differ because their averaging weights
differ. Top-3 accuracy describes shortlist coverage, not top-1 correctness.

### 4.1 Cross-entropy and probability scores

Evaluation cross-entropy is the mean $-\log(p_{i,y_i})$; it penalizes confident
errors more heavily than accuracy. The multiclass Brier score [9] in
`src/satscene/metrics.py` is

$$
\mathrm{Brier}=\frac{1}{N}\sum_{i=1}^{N}\sum_{c=0}^{K-1}
\left(p_{ic}-\mathbf{1}[y_i=c]\right)^2.
$$

For this sum-over-classes convention, the score lies between 0 and 2, with
lower values better. Top-label expected calibration error (ECE) uses `M = 10`
equal-width confidence bins, following the standard binned approximation [10]. With $q_i=\max_c p_{ic}$,
$a_i=\mathbf{1}[\hat{y}_i=y_i]$, and $B_m$ the examples whose $q_i$ fall in bin $m$,

$$
\mathrm{ECE}=\sum_{m=1}^{M}\frac{|B_m|}{N}
\left|\operatorname{acc}(B_m)-\operatorname{conf}(B_m)\right|,
$$

$$
\operatorname{acc}(B_m)=\frac{1}{|B_m|}\sum_{i\in B_m}a_i,\qquad
\operatorname{conf}(B_m)=\frac{1}{|B_m|}\sum_{i\in B_m}q_i.
$$

ECE is bin-dependent and can hide within-bin errors; it should be read together
with a reliability diagram and a proper score such as cross-entropy or Brier.
The model comparison's ECE values are descriptive estimates on only 200 examples.

### 4.2 Wilson interval for accuracy

For $k$ correct predictions among $n$ examples, $\hat{p}=k/n$. The two-sided
Wilson score interval [11,12] uses normal quantile $z$ and is

$$
d=1+\frac{z^2}{n},\qquad
\mathrm{centre}=\frac{\hat{p}+z^2/(2n)}{d},\qquad
\mathrm{half}=\frac{z\sqrt{\hat{p}(1-\hat{p})/n+z^2/(4n^2)}}{d},
$$

$$
\mathrm{CI}_{1-\alpha}=[\mathrm{centre}-\mathrm{half},\;
\mathrm{centre}+\mathrm{half}].
$$

For ConvNeXtV2-Tiny, $k=160$, $n=200$, $\hat{p}=0.800$, and $z=1.96$, giving
approximately $[0.7391,0.8495]$. For SwinV2-Tiny, $k=159$, $\hat{p}=0.795$,
giving $[0.7337,0.8451]$. These are per-model binomial intervals. Because the
same validation images were used for both models, comparing them formally calls
for a paired analysis (for example, paired bootstrap over examples or McNemar's
test on discordant predictions), not subtraction of independent intervals.
Neither method addresses retraining variability, dataset shift, or label noise.

## 5. Numerical reading of the archived comparison

The comparison CSV gives the following validation values ($N=200$):

| Model | Correct / 200 | Accuracy | Macro-F1 | Balanced accuracy | Top-3 | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ConvNeXtV2-Tiny | 160 | 0.800 | 0.7881 | 0.7876 | 0.940 | 0.1176 |
| SwinV2-Tiny | 159 | 0.795 | 0.7817 | 0.7770 | 0.935 | 0.1288 |
| DINOv2 ViT-S/14 | 76 | 0.380 | 0.3049 | 0.3441 | 0.690 | 0.1027 |

The direct accuracy difference for ConvNeXtV2 vs SwinV2 is
$(160-159)/200=0.005$, i.e. one image or 0.5 percentage points. The macro-F1
difference is about $0.7881-0.7817=0.0063$. These small single-run gaps are
not strong evidence of a stable ranking. DINOv2 has lower recorded ECE than the
other two, but its accuracy and macro-F1 are much lower; calibration alone does
not measure classification quality. None of these rows is a held-out test
comparison, and no across-seed standard deviation is available.

## 6. Shape and interaction summary

Spatial sizes below follow the selected input sizes and canonical model
configuration. $H\times W$ counts spatial positions (or tokens), not channels. Channel
width changes by stage and is omitted here because only spatial interaction
counts are being compared.

| Backbone | Input and tokenization | Hierarchy / depth | Attention interaction pattern |
| --- | --- | --- | --- |
| ConvNeXtV2-Tiny | $224\times224$ RGB; $4\times4$ stem gives $56\times56$ features | $56^2\to28^2\to14^2\to7^2$; depths 3/3/9/3 | Convolutional neighborhoods; not self-attention |
| SwinV2-Tiny | $256\times256$ RGB; $4\times4$ patch embedding gives $64\times64$ tokens | $64^2\to32^2\to16^2\to8^2$; depths 2/2/6/2 | Windowed attention, shifted between blocks; window size 8 |
| DINOv2 ViT-S/14 | $224\times224$ RGB; $14\times14$ patches give $16\times16+\mathrm{CLS}=257$ tokens | 12 transformer blocks, fixed token grid | Global all-pairs attention in each block |

The architectures have different channel dimensions, operator counts, and
input sizes, so token-score counts alone cannot rank runtime or memory. For
reproducible efficiency comparisons, report measured peak memory, throughput,
latency, parameter count, and multiply-accumulate/FLOP convention on the same
hardware and batch size.

## References and further reading

### Architecture and representation learning

1. Liu, Z. et al. (2022). [A ConvNet for the 2020s](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html). CVPR.
2. Woo, S. et al. (2023). [ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://openaccess.thecvf.com/content/CVPR2023/html/Woo_ConvNeXt_V2_Co-Designing_and_Scaling_ConvNets_With_Masked_Autoencoders_CVPR_2023_paper.html). CVPR.
3. Liu, Z. et al. (2021). [Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.html). ICCV.
4. Liu, Z. et al. (2022). [Swin Transformer V2: Scaling Up Capacity and Resolution](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Swin_Transformer_V2_Scaling_Up_Capacity_and_Resolution_CVPR_2022_paper.html). CVPR.
5. Dosovitskiy, A. et al. (2021). [An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale](https://openreview.net/forum?id=YicbFdNTTy). ICLR.
6. Oquab, M. et al. (2023). [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193). arXiv:2304.07193.

### Optimization, metrics, and statistical methodology

7. Loshchilov, I. & Hutter, F. (2019). [Decoupled Weight Decay Regularization](https://openreview.net/forum?id=Bkg6RiCqY7). ICLR.
8. Loshchilov, I. & Hutter, F. (2017). [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983). arXiv:1608.03983.
9. Brier, G. W. (1950). [Verification of Forecasts Expressed in Terms of Probability](https://doi.org/10.1175/1520-0493(1950)078%3C0001:VOFEIT%3E2.0.CO;2). Monthly Weather Review.
10. Guo, C. et al. (2017). [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html). ICML / PMLR 70.
11. Wilson, E. B. (1927). [Probable Inference, the Law of Succession, and Statistical Inference](https://doi.org/10.1080/01621459.1927.10502953). Journal of the American Statistical Association.
12. Newcombe, R. G. (1998). [Two-sided confidence intervals for the single proportion: comparison of seven methods](https://doi.org/10.1002/(SICI)1097-0258(19980430)17:8%3C857::AID-SIM777%3E3.0.CO;2-E). Statistics in Medicine.

### Remote sensing, implementation, and data provenance

13. Cheng, G. et al. (2023). [Remote sensing scene classification under scarcity of labelled samples—A survey of the state-of-the-arts](https://doi.org/10.1016/j.cageo.2022.105295). Computers & Geosciences.
14. Xia, G.-S. et al. (2017). [AID: A Benchmark Data Set for Performance Evaluation of Aerial Scene Classification](https://doi.org/10.1109/TGRS.2017.2685945). IEEE TGRS.
15. Cheng, G. et al. (2017). [Remote Sensing Image Scene Classification: Benchmark and State of the Art](https://arxiv.org/abs/1703.00121) (NWPU-RESISC45).
16. Helber, P. et al. (2019). [EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification](https://arxiv.org/abs/1709.00029). IEEE JSTARS.
17. Sumbul, G. et al. (2019). [BigEarthNet: A Large-Scale Benchmark Archive for Remote Sensing Image Understanding](https://arxiv.org/abs/1902.06148). IGARSS.
18. [PyTorch `CrossEntropyLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html), [AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html), and [CosineAnnealingLR](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html) API documentation.
19. [PyTorch Image Models (`timm`) model API](https://huggingface.co/docs/timm/reference/models).
20. [Remote Sensing Satellite Images dataset](https://www.kaggle.com/datasets/umeradnaan/remote-sensing-satellite-images) and [Roboflow YOLOv8 export](https://universe.roboflow.com/image-captioning-for-remote-sensingdata-0zids/remote-sensing-data/dataset/2).

Remote-sensing scene-classification surveys describe the limited-label regime
as a recurring methodological challenge [13]. AID, NWPU-RESISC45, EuroSAT, and
BigEarthNet are useful alternative benchmarks [14–17], but their sensors,
labels, and split designs differ from this assignment dataset. They are not
directly comparable to SatScene's validation scores without a reproducible
protocol and label mapping.

This bibliography supports the mathematical background. Dataset licenses,
pretrained-weight licenses, and exact model-card/pretraining provenance should be
checked independently before redistribution or operational use.
