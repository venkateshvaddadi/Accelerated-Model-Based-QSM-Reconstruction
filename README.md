
# Accelerated QSM Reconstruction Engine

This repository hosts production-ready, modularized PyTorch implementations for advanced Model-Based Iterative Reconstruction (MBIR) algorithms dedicated to **Quantitative Susceptibility Mapping (QSM)**. 

The framework features two distinct mathematical acceleration structures to resolve the ill-posed dipole inversion problem:
1. **Reduced Rank Extrapolation (RRE)** physics-informed variants (adapted from Awasthi et al., JBO 2018).
2. **Nesterov Momentum-Accelerated** proximal gradient descent architectures.

Both methods seamlessly pair classical forward QSM physics operators with optional neural network image refiners (such as 3D U-Nets or WideResNets) to balance rigid data-fidelity bounds with strong generative image priors.

---

## 📂 Repository Structure

```text
├── modules/
│   ├── metrics.py          # Validation metrics (PSNR, RMSE, HFEN, SSIM)
│   └── model_from_DIP.py   # Alternative Deep Image Prior network baselines
├── models.py               # Deep denoiser structures & core unrolled QSM physics networks
├── utils.py                # Analytical 3D dipole kernel generators, Sobel losses, & plots
├── main.py                 # Primary execution orchestration script (Loops & evaluations)
└── .gitignore              # Out-of-commit exclusion policies for datasets and logs
