Here is a **clean, professional README-style version** suitable for GitHub:

---

# MA-MBIR-QSM

**Momentum-Accelerated Model-Based QSM Reconstruction**

---

## Overview

MA-MBIR-QSM is an iterative reconstruction algorithm for Quantitative Susceptibility Mapping (QSM).
It combines a physics-based forward model with momentum acceleration (FISTA-style) to improve convergence speed and reconstruction quality.

---

## Inputs

* **`y`** : Local field
* **`D`** : Dipole kernel
* **`F`** : Fourier transform operator
* **`K`** : Number of iterations
* **`L`** : Step size

---

## Output

* **`χ`** : Reconstructed susceptibility map

---

## Algorithm

```
Initialize:
    χ₀ = 0
    χ₋₁ = 0
    θ₀ = 1

For k = 0 to K-1:

    Momentum update:
        θₖ₊₁ = (1 + sqrt(1 + 4θₖ²)) / 2
        mₖ = (θₖ - 1) / θₖ₊₁

    Extrapolation:
        χ̂ₖ = χₖ + mₖ (χₖ - χₖ₋₁)

    Forward model:
        φ(χ̂ₖ) = Fᴴ ( D · F(χ̂ₖ) )

    Residual:
        rₖ = φ(χ̂ₖ) - y

    Gradient:
        gₖ = Fᴴ ( D · F(rₖ) )

    Update:
        χₖ₊₁ = χ̂ₖ - (1 / L) gₖ

    Update variables:
        χₖ₋₁ = χₖ
        χₖ = χₖ₊₁
        θₖ = θₖ₊₁

Return χ_K
```

---

## Key Components

* **Forward Model**:
  φ(χ) = Fᴴ D F(χ)

* **Momentum Acceleration**:
  Uses FISTA-style update to accelerate convergence.

* **Gradient Update**:
  Computed using the adjoint forward model.

---

## Notes

* `F` denotes Fourier transform
* `Fᴴ` denotes inverse (Hermitian) Fourier transform
* `D` is applied in k-space (element-wise multiplication)
* Suitable for model-based reconstruction with iterative refinement


