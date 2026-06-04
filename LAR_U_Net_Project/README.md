# LAR-U-Net: Lightweight Attentive Residual U-Net for Image Denoising

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced, lightweight image restoration architecture engineered for grayscale image denoising. **LAR-U-Net** structurally combines the spatial localization precision of a symmetric U-Net backbone, the adaptive feature refinement of a Convolutional Block Attention Module (CBAM), and a residual noise-mapping paradigm. Additionally, it leverages a handcrafted **Gradient-Augmented Processing** front-end to strictly preserve structural boundaries and high-frequency edge dynamics.

---

## 🛠️ Architecture Overview

The core challenge of discriminative denoisers is avoiding the over-smoothing of critical high-frequency components (e.g., fine textures, sharp edges), which degrades performance on downstream vision tasks. LAR-U-Net implements a four-tiered architectural defense against this:

1. **Gradient-Augmented Processing (Front-End):** Instead of processing the raw noisy image in isolation, the framework passes the input through horizontal ($G_x$) and vertical ($G_y$) Sobel filters. The resulting gradient magnitude map ($G$) is concatenated directly along the channel dimension, forcing early layers to prioritize edge maps.
2. **Symmetric U-Net Backbone:** Utilizes a standard encoder-decoder structure linked via skip connections. It balances model capacity and performance metrics using optimal $3 \times 3$ convolutional configurations.
3. **Dual-Axis Attention Gates (CBAM):** Embedded directly inside each encoder block to dynamically weigh feature importance:
   * **Channel Attention:** Compresses spatial dimensions using adaptive global average pooling to discover cross-channel feature dependencies (signal vs. noise variants).
   * **Spatial Attention:** Concatonates average and max pooling mappings across the channel dimension, applying a localized $7 \times 7$ kernel to generate an explicit spatial mask highlighting areas where noise is most destructive.
4. **Residual Noise Learning Paradigm:** Following the discriminative paradigm of DnCNN, the network does not attempt to construct a clean image directly. Instead, it maps the underlying *Noise Map* $R(y)$.

---

## 📐 Mathematical Formulation

### 1. Image Degradation & Reconstruction Model
The noisy observation $y$ is assumed to follow an additive noise distribution:
$$y = x + v$$

Where $x$ represents the true latent clean image and $v$ represents the corrupting additive white Gaussian noise (AWGN) component with standard deviation $\sigma$. The final reconstructed image $\hat{x}$ is obtained via explicit subtraction:
$$\hat{x} = y - R(y)$$

### 2. Front-End Edge Gradient Extraction
Using discrete Sobel operators, spatial gradients are evaluated:
$$G = \sqrt{G_x^2 + G_y^2}$$

### 3. Optimization Criterion
The system optimizes network parameterizations ($W$) by minimizing the Mean Squared Error (MSE) loss regularized explicitly over the targeted residual noise map:
$$\mathcal{L}(W) = \frac{1}{2N} \sum_{i=1}^{N} \| R(y_i; W) - (y_i - x_i) \|_F^2$$

---

## 📊 Experimental Setup & Performance Metrics

### Training Configuration
* **Optimizer:** Adam
* **Loss Function:** Mean Squared Error (MSE)
* **Epochs:** 10
* **Batch Size:** 8
* **Target Noise Setup:** AWGN ($\sigma = 25$)
* **Core Dataset:** Berkeley Segmentation Dataset 500 (BSDS500) — explicit disjoint splits.
* **Robustness Suite:** Multi Noises for Image Denoising Dataset (Evaluated against Speckle, Poisson, Multiplicative, JPEG artifacts, Quantization, and Salt & Pepper).

### Key Empirical Results
The model exhibits rapid convergence and strong numeric optimization characteristics on Gaussian distributions:
* **Minimum Training Loss:** `0.00315` (MSE)
* **Minimum Validation Loss:** `0.00327` (MSE)
* **Test Dataset Mean Absolute Error (MAE):** `0.04253`
* **Average Suite PSNR:** `19.9205 dB`
* **Peak Structural Improvement:** Exceeded **+7 dB** on highly structured natural landscape samples.

#### Epoch-Wise Optimization Progress (Table I)
| Epoch | Train Loss (MSE) | Validation Loss (MSE) |
| :---: | :--------------: | :------------------: |
| 1     | 0.22759          | 0.24682              |
| 2     | 0.00983          | 0.00924              |
| 5     | 0.00415          | 0.00414              |
| 10    | 0.00315          | 0.00327              |

#### Localized Sample Evaluation Breakdown (Table II)
| Evaluation Parameter | Sample 1 | Sample 2 |
| :------------------- | :------: | :------: |
| **Initial PSNR** | 20.21 dB | 20.19 dB |
| **Denoised PSNR** | 28.01 dB | 27.36 dB |
| **Net PSNR Gain** | **+7.79 dB** | **+7.18 dB** |
| **Initial MAE** | 0.0779   | 0.0783   |
| **Denoised MAE** | 0.0306   | 0.0331   |
| **Net MAE Reduction**| **-0.0473** | **-0.0452** |

> **Architectural Note on Generalization Boundaries:** While LAR-U-Net yields high PSNR gains on Salt & Pepper and Multiplicative metrics during quantitative data checks, visual inspection demonstrates that its denoising performance is intensely localized toward Gaussian distributions. This is primarily because residual mappings and Batch Normalization layers operate symmetrically along Gaussian probability densities, allowing them to mutually regularize each other during AWGN optimization steps.

---

## 🚀 Getting Started

### 1. Installation & Dependency Tracking
Clone the repository and ensure your local Python environment has the matching deep learning packages installed:
```bash
git clone [https://github.com/your-username/LAR-U-Net.git](https://github.com/your-username/LAR-U-Net.git)
cd LAR_U_Net_Project
