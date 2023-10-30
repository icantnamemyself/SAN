### SAN (NeurIPS 2023)

---

This repo is the official Pytorch implementation of our NeurIPS 2023 paper: Adaptive Normalization for Non-stationary Time Series Forecasting: A Temporal Slice Perspective.

#### Introduction

Time series data suffer from a non-stationary issue where the statistical properties or the distributions of the data vary rapidly over time. We further argue that the distribution is inconsistent across compact time slices and such inconsistency is not just on a per-instance basis. To alleviate the impact of such property, we propose a model-agnostic normalization framework named SAN. SAN models the non-stationarity in the fine-grained  temporal slices and explicitly learn to estimate future distributions, simplifying the the non-stationary forecasting task through divide and conquer.

![framework](figs\framework.png)

We conduct comparison experiments on 9 widely used datasets with mainstream forecasting backbones, we also compare the performance of SAN and other plug-and-play non-stationary methods.

Multivariate forecasting results:

![multivariate](figs\multivariate.png)

Comparison with other plug-and-play non-stationary methods:

![compare](figs/compare.png)

#### Usage

##### Environment and dataset setup

```bash
pip install -r requirements.txt
mkdir datasets
```

All the 9 datasets are available at the [Google Driver]([Autoformer - Google 云端硬盘](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy)) provided from Autoformer. Many thanks to their efforts and devotion!

##### Running

We provide ready-to-use scripts for SAN enhanced backbone models.

```bash
sh run_linear.sh # scripts for DLinear
sh run_trms.sh # scripts for Transformers (Autoformer/FEDformer/Informer/Transformer)
```

##### Tuning



#### Acknowledgement

This repo is built on the pioneer works. We appreciate the following GitHub repos a lot for their valuable code base or datasets:

[Informer]([zhouhaoyi/Informer2020: The GitHub repository for the paper "Informer" accepted by AAAI 2021.](https://github.com/zhouhaoyi/Informer2020))

[Autoformer]([thuml/Autoformer: About Code release for "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting" (NeurIPS 2021), https://arxiv.org/abs/2106.13008 (github.com)](https://github.com/thuml/Autoformer))

[DLinear]([cure-lab/LTSF-Linear: [AAAI-23 Oral\] Official implementation of the paper "Are Transformers Effective for Time Series Forecasting?" (github.com)](https://github.com/cure-lab/LTSF-Linear))

#### Citation

TBD
