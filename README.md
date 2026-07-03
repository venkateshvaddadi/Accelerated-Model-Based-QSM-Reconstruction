# Accelerated Model-Based Quantitative Susceptibility Mapping Using Momentum and Vector Extrapolation

Official PyTorch implementation of the paper:

> **Accelerated Model-Based Quantitative Susceptibility Mapping Using Momentum and Vector Extrapolation**

---

## Overview

This repository provides two accelerated optimization algorithms for **Model-Based Quantitative Susceptibility Mapping (QSM)** reconstruction.

Unlike conventional approaches that improve QSM reconstruction through sophisticated regularization models or deep learning priors, the proposed methods focus on **accelerating the optimization process itself** while preserving the underlying physics-based dipole inversion formulation.

The repository includes implementations of:

- **Momentum-Accelerated Model-Based QSM (MA-Mo-QSM)**
- **Reduced Rank Extrapolation-Based QSM (RRE-QSM)**

Both methods solve the QSM inverse problem by enforcing data consistency through the dipole forward model while accelerating convergence using established optimization techniques.

## Reconstruction Frameworks

### Momentum-Accelerated Model-Based QSM (MA-Mo-QSM)

MA-Mo-QSM incorporates **Nesterov momentum acceleration** into iterative model-based QSM reconstruction.

Each iteration consists of

1. Momentum extrapolation
2. Physics-based data consistency update
3. Gradient descent reconstruction

---

### Reduced Rank Extrapolation QSM (RRE-QSM)

RRE-QSM accelerates convergence by combining multiple previous iterates using **Reduced Rank Extrapolation (RRE)**.

Each optimization cycle consists of

1. Model-based gradient descent
2. Construction of iterate differences
3. QR-based coefficient estimation
4. Extrapolated susceptibility update

---

## Repository Structure

```
Accelerated-QSM/
│
├── MA_MO_QSM_main.py          # Main script for Momentum-Accelerated QSM
├── RRE_QSM_main.py            # Main script for Reduced Rank Extrapolation QSM
├── models.py                  # Model definitions for MA-Mo-QSM and RRE-QSM
├── utils.py                   # QSM utilities, losses, dipole kernel, visualization
├── modules/
│   └── metrics.py             # Quantitative evaluation metrics
│
├── savedModels/               # Reconstruction outputs (generated automatically)
│
└── README.md
```

---

## Requirements

The implementation has been tested using

- Python 3.12
- PyTorch 2.4+
- CUDA 12.x

Install the required packages using

```bash
pip install torch numpy scipy pandas matplotlib
```

---

## Dataset

The implementation expects the following dataset structure:

```
QSM_data/
└── data_for_experiments/
    └── given_data/
        └── raw_data_names_modified/
            ├── patient_1/
            ├── patient_2/
            ├── ...
            └── patient_12/
```

Each patient folder should contain

```
phs1.mat
phs2.mat
...

msk1.mat
...

mag1.mat
...

cos1.mat
...
```

where

- **phs** : Local field map
- **msk** : Brain mask
- **mag** : Magnitude image
- **cos** : COSMOS susceptibility map (ground truth)

---

# Running MA-Mo-QSM

Run

```bash
python MA_MO_QSM_main.py
```

This script performs

- Momentum-accelerated model-based reconstruction
- Quantitative evaluation
- Saves reconstructed susceptibility maps
- Exports reconstruction metrics as CSV

---

# Running RRE-QSM

Run

```bash
python RRE_QSM_main.py
```

This script performs

- Reduced Rank Extrapolation accelerated reconstruction
- Quantitative evaluation
- Saves reconstructed susceptibility maps
- Exports reconstruction metrics as CSV

---


---

## Output

For each experiment the repository generates

```
savedModels/
└── Experiment_Name/
    ├── patient_1_orientation_1.mat
    ├── patient_1_orientation_2.mat
    ├── ...
    ├── momentum_qsm_results.csv
    └── qsm_results.csv
```

---

## Evaluation Metrics

The following metrics are computed automatically.

- SSIM
- PSNR
- RMSE
- HFEN

---


---

## License

This repository is released for academic research purposes.

---

## venkateshvaddadi254@gmail.com

**Vaddadi Venkatesh**

Department of Computational and Data Sciences

Indian Institute of Science (IISc)

Bengaluru, India

Email: *your_email_here*
