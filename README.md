# ROFANet

**ROFANet: Reliable Occlusion-aware Fine-grained Alignment Network for Text-based Occluded Person Re-Identification**

ROFANet is a framework for **text-based occluded person re-identification (TO-ReID)**. Given a natural-language pedestrian description, it retrieves matching person images from a gallery under occluded conditions.

Built on CLIP, ROFANet combines semantic-aware visibility estimation with part-aware semantic guidance to improve fine-grained image–text alignment. A visibility- and reliability-aware weighting mechanism helps reduce interference from occluded or semantically ambiguous regions.

## Overview

The framework of ROFANet is illustrated in the following figure:

<img width="3917" height="2083" alt="framework-new-cropped" src="https://github.com/user-attachments/assets/d571e18d-af62-4f3a-baab-bcb4d3e6cfbb" />

ROFANet includes two main modules and a weighted alignment mechanism:

1. **Semantic-aware Visibility Estimation (SVE)**

   SVE uses cross-modal attention to derive region-level semantic attributions, including body and background, and estimates region visibility through response discrepancies. These estimates help suppress interference from occluded or semantically ambiguous regions.

2. **Part-aware Semantic Guidance (PSG)**

   PSG aggregates image patches and text tokens into part-level representations under textual semantic guidance, enabling fine-grained alignment between visual regions and textual descriptions.

3. **Visibility- and Reliability-aware Alignment**

   During feature alignment, ROFANet incorporates region visibility and reliability pseudo-labels derived from RDE to prioritize reliable image–text pairs and visible regions for cross-modal matching.

## Requirements

The experiments described in the paper use:

```text
Operating system: Ubuntu 22.04
Python:           3.8
Framework:        PyTorch
GPU:              One NVIDIA RTX 3090
```

For environment setup and dependency installation, please refer to the [RDE repository](https://github.com/QinYang79/RDE).

## Supported Datasets

ROFANet is evaluated on three public TO-ReID benchmarks:

```text
Occluded-CUHK-PEDES
Occluded-ICFG-PEDES
Occluded-RSTPReid
```

For dataset download links and occlusion generation instructions, please refer to the **Data preparation** section of the [MGCC repository](https://github.com/littlexinyi/MGCC).

Please follow the evaluation protocol described in our paper when preparing the datasets.

## Acknowledgments

We thank the authors of [RDE](https://github.com/QinYang79/RDE), [MGCC](https://github.com/littlexinyi/MGCC), and [CLIP](https://github.com/openai/CLIP), as well as the creators of the datasets used in this work, for making their resources publicly available.
