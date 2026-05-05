# LAR-U-Net

This project implements the **Lightweight Attentive Residual U-Net (LAR-U-Net)**, an optimized architecture for image denoising. The model was trained and evaluated using the **BSDS500** dataset and a variety of noise types to test its robustness and structural preservation capabilities.

Detailed implementation and experiments can be found in the notebook: `LAR_UNet_using_BSDS500_and_Multi_Noises_Final_edit.ipynb`.

---

## 1. Model Architecture

The LAR-U-Net is a hybrid architecture that integrates the spatial precision of a U-Net with the global receptive field of Vision Transformers and the refinement of Attention Modules.

### A. Gradient-Augmented Processing
Unlike standard models that input only raw pixels, LAR-U-Net utilizes **Sobel Filters** to calculate horizontal ($G_x$) and vertical ($G_y$) gradients. This highlights edges before the first convolution.
*   **Gradient Magnitude ($G$):**
$$G = \sqrt{G_x^2 + G_y^2}$$

### B. Convolutional Block Attention Module (CBAM)
Each encoder layer includes a CBAM block to focus on relevant features across channels and spatial locations.
*   **Refined Feature ($F''$):**
$$F' = M_c(F) \otimes F$$
$$F'' = M_s(F') \otimes F'$$
*(Where M_c is the Channel Attention and M_s is the Spatial Attention).*

### C. Lightweight ViT Bottleneck
At the lowest resolution, a **Lightweight Vision Transformer (ViT)** block is implemented. This allows the model to capture global noise trends and repetitive patterns that local convolutions might overlook.

### D. Residual Learning
The model predicts a **Noise Map** $\mathcal{R}(y)$ rather than the clean image. The final denoised output $\hat{x}$ is obtained via:
$$\hat{x} = y - \mathcal{R}(y)$$

---

## 2. Technical Comparison

| Feature | LAR-U-Net | DnCNN | SwinIR |
| :--- | :--- | :--- | :--- |
| **Backbone** | U-Net (Encoder-Decoder) | Plain Deep CNN | Swin Transformer |
| **Global Context** | ViT Bottleneck | None (Local only) | High (Full Transformer) |
| **Complexity** | Lightweight / Optimized | Very Lightweight | Computationally Heavy |
| **Edge Retention**| High (Gradient Augment) | Medium (Over-smooths) | Very High |

---

## 3. Mathematical Metrics

The performance is measured using **Mean Squared Error (MSE)** and **Peak Signal-to-Noise Ratio (PSNR)**.

*   **Mean Squared Error (MSE):** Measures the average squared difference between the clean image $I$ and the reconstructed image $K$.
$$MSE = \frac{1}{mn} \sum_{i=0}^{m-1} \sum_{j=0}^{n-1} [I(i,j) - K(i,j)]^2$$

*   **Peak Signal-to-Noise Ratio (PSNR):** Represents the ratio between the maximum possible power of a signal and the corrupting noise, expressed in decibels (dB).
$$PSNR = 10 \cdot \log_{10} \left( \frac{MAX_I^2}{MSE} \right)$$

---

## 4. Performance and Training

The model was trained for 10 epochs using the Adam optimizer and an MSE loss function. 
*   **Training Artifacts:** The trained model parameters are saved in `lar_unet_weights.pth`.
*   **Optimization Progress:** See `loss_curve.png` for a visualization of the training and validation loss convergence.

### Results
*   **Test MAE:** 0.03375
*   **Visualization:** The model achieves a significant PSNR improvement across various noise types (Salt & Pepper, Speckle, Gaussian, etc.). 
*   Refer to `psnr_chart.png` for the quantitative improvement data.
*   Refer to `multi_noise_comparison.jpg` for side-by-side visual results of the denoising process.

---

## 5. Usage

To replicate the results or utilize the model for inference:
1.  **Environment:** Ensure you have PyTorch, Torchvision, Matplotlib, and PIL installed.
2.  **Weights:** Load the pre-trained weights from `lar_unet_weights.pth`.
3.  **Notebook:** Run the cells in `LAR_UNet_using_BSDS500_and_Multi_Noises_Final_edit.ipynb` for full data preprocessing and evaluation workflows.
