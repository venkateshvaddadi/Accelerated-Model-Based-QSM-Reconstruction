#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 11:01:39 2026
@author: venkatesh
"""

# --- Standard Library & OS ---
import os
import time
from datetime import datetime
from pathlib import Path

# --- Scientific Computing ---
import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt

# --- PyTorch Core ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

# --- Custom Metrics ---
from modules.metrics import compute_psnr, compute_rmse, compute_hfen, compute_ssim_numpy_version_2

def dipole_kernel(matrix_size, voxel_size, B0_dir=[0, 0, 1]):
    [Y, X, Z] = np.meshgrid(
        np.linspace(-int(matrix_size[1]/2), int(matrix_size[1]/2)-1, matrix_size[1]),
        np.linspace(-int(matrix_size[0]/2), int(matrix_size[0]/2)-1, matrix_size[0]),
        np.linspace(-int(matrix_size[2]/2), int(matrix_size[2]/2)-1, matrix_size[2])
    )
    
    X = X / (matrix_size[0]) * voxel_size[0]
    Y = Y / (matrix_size[1]) * voxel_size[1]
    Z = Z / (matrix_size[2]) * voxel_size[2]
    
    numerator = np.square(X * B0_dir[0] + Y * B0_dir[1] + Z * B0_dir[2])
    denominator = np.square(X) + np.square(Y) + np.square(Z) + np.finfo(float).eps
    D = 1/3 - np.divide(numerator, denominator)
    D = np.where(np.isnan(D), 0, D)

    D = np.roll(D, int(np.floor(matrix_size[0]/2)), axis=0)
    D = np.roll(D, int(np.floor(matrix_size[1]/2)), axis=1)
    D = np.roll(D, int(np.floor(matrix_size[2]/2)), axis=2)
    
    D = np.float32(D)
    return torch.tensor(D).unsqueeze(dim=0)

device_id = 0
matrix_size = [176, 176, 160]
voxel_size = [1, 1, 1]
dk = dipole_kernel(matrix_size, voxel_size, B0_dir=[0, 0, 1]).cuda(device_id)

class MomentumNetQSM_GradL2(nn.Module):
    def __init__(self, num_iters=9, rho=1.0, lambda_reg=0.01):
        super().__init__()
        self.num_iters = num_iters
        self.rho = rho 
        self.lambda_reg = lambda_reg  # Weight for gradient L2 regularization
        self.L = nn.Parameter(torch.tensor(1.0))        

    def forward(self, local_field, dipole_kernel):
        x_curr = torch.zeros_like(local_field)
        x_prev = torch.zeros_like(local_field)
        theta_prev = 1.0
        
        for i in range(self.num_iters):
            # --- 1. Extrapolation Module (Momentum) ---
            theta_curr = (1 + (1 + 4 * theta_prev**2)**0.5) / 2
            m = (theta_prev - 1) / theta_curr
            x_hat = x_curr + m * (x_curr - x_prev)
            
            # --- 2. MBIR Module with L2 Gradient Regularization ---
            # Data consistency residual: D * x_hat - local_field
            residual = self.dipole_conv(x_hat, dipole_kernel) - local_field
            data_grad = self.dipole_conv(residual, dipole_kernel) 
            
            # Compute 3D Spatial Gradient Regularization (Laplacian approximation via finite differences)
            reg_grad = self.compute_spatial_gradient_penalty(x_hat)
            
            # Total Analytical Gradient: D^T(Dx - y) - lambda * Laplacian(x)
            grad = data_grad + (self.lambda_reg * reg_grad)
            
            # Gradient descent step
            gradient_step = x_hat - (1.0 / self.L) * grad 
            
            # Apply Relaxation Parameter (rho)
            x_next = x_curr + self.rho * (gradient_step - x_curr)
            
            # Update states
            x_prev = x_curr
            x_curr = x_next
            theta_prev = theta_curr
            
        return x_curr

    def dipole_conv(self, x, kernel):
        return torch.real(torch.fft.ifftn(torch.fft.fftn(x) * kernel))

    def compute_spatial_gradient_penalty(self, x):
        """
        Computes the analytical derivative of the L2 norm of spatial gradients 
        (acts similarly to the negative Laplacian operator).
        """
        # Finite differences along X, Y, Z dimensions
        dx = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
        dy = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
        dz = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
        
        # Pad back to original dimensions to match shape
        dx_pad = F.pad(dx, (0, 0, 0, 0, 0, 1), mode='constant', value=0)
        dy_pad = F.pad(dy, (0, 0, 0, 1, 0, 0), mode='constant', value=0)
        dz_pad = F.pad(dz, (0, 1, 0, 0, 0, 0), mode='constant', value=0)
        
        # Divergence of gradients gives the Laplacian-like penalty term
        ddx = dx_pad[:, :, 1:, :, :] - dx_pad[:, :, :-1, :, :]
        ddy = dy_pad[:, :, :, 1:, :] - dy_pad[:, :, :, :-1, :]
        ddz = dz_pad[:, :, :, :, 1:] - dz_pad[:, :, :, :, :-1]
        
        # Pad divergence back to full volume shape [B, C, D, H, W]
        ddx_full = F.pad(ddx, (0, 0, 0, 0, 1, 0), mode='constant', value=0)
        ddy_full = F.pad(ddy, (0, 0, 1, 0, 0, 0), mode='constant', value=0)
        ddz_full = F.pad(ddz, (1, 0, 0, 0, 0, 0), mode='constant', value=0)
        
        return -(ddx_full + ddy_full + ddz_full)

# ==========================================
# Experiment Setup
# ==========================================
base_dir = "savedModels/MOMENTUM_QSM_GRADL2_EXPERIMENTS"
model_name = "MomentumQSM_GradL2"
num_iters = 10
rho = 1.0
lambda_reg = 0.01

timestamp = datetime.now().strftime("%b_%d_%H_%M_%S")
exp_name = f"{model_name}___K_{num_iters}_lam_{lambda_reg}"
exp_dir = os.path.join(base_dir, exp_name)
os.makedirs(exp_dir, exist_ok=True)

print("\n" + "="*50)
print(f"🚀 STARTING EXPERIMENT: {model_name}")
print("="*50)
print(f"📂 Save Directory:  {exp_dir}")
print(f"🕒 Timestamp:       {timestamp}")
print(f"🔄 Iterations (K):  {num_iters}")
print(f"⚖️ Grad L2 Reg (λ): {lambda_reg}")
print(f"💻 Device:          CUDA:{device_id}")
print("="*50 + "\n")

model = MomentumNetQSM_GradL2(num_iters=num_iters, rho=rho, lambda_reg=lambda_reg).to(device_id)
criterion = nn.MSELoss()

batch_size = 1
results = []
patients_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] 
orientations = [1, 2, 3, 4, 5]


raw_data_path = '../../QSM_data/data_for_experiments/given_data/raw_data_names_modified/'

for i in patients_list:
    for j in orientations:
        phs = scipy.io.loadmat(f"{raw_data_path}/patient_{i}/phs{j}.mat")['phs']
        msk = scipy.io.loadmat(f"{raw_data_path}/patient_{i}/msk{j}.mat")['msk']
        mag = scipy.io.loadmat(f"{raw_data_path}/patient_{i}/mag{j}.mat")['mag']
        
        phs = torch.tensor(phs).float().unsqueeze(0).unsqueeze(0).cuda(device_id)
        msk = torch.tensor(msk).float().unsqueeze(0).unsqueeze(0).cuda(device_id)
        mag = torch.tensor(mag).float().unsqueeze(0).unsqueeze(0).cuda(device_id)
        
        dk_rep = dk.repeat(batch_size, 1, 1, 1, 1)

        # Forward pass (Fast GPU Execution)
        start_time = time.time()
        chi_pred = model(phs, dk_rep)
        print(f"Patient {i} Orientation {j} processed in {time.time() - start_time:.4f}s")

        phi_pred = torch.fft.ifftn(dk_rep * torch.fft.fftn(chi_pred, dim=[2,3,4]), dim=[2,3,4])
        loss_fidelity = torch.mean(torch.abs(phi_pred - phs))
        loss = loss_fidelity
        
        x_k_cpu = (chi_pred.real.detach().cpu().numpy()) * (msk.detach().cpu().numpy())
        x_k_cpu = np.squeeze(x_k_cpu)

        # Compute Metrics
        sus = scipy.io.loadmat(f"{raw_data_path}/patient_{i}/cos{j}.mat")['cos']
        psnr_val = compute_psnr(x_k_cpu, sus)
        rmse_val = compute_rmse(x_k_cpu, sus)
        hfen_val = compute_hfen(x_k_cpu, sus)
        ssim_val = compute_ssim_numpy_version_2(x_k_cpu, sus)

        print({
            'patient': i,
            'orientation': j,
            'SSIM': f'{ssim_val:.4f}',
            'PSNR': f'{psnr_val:.2f}',
            'RMSE': f'{rmse_val:.4f}',
            'HFEN': f'{hfen_val:.4f}',
        })

        results.append({
            'patient': i,
            'orientation': j,
            'ssim': ssim_val,
            'psnr': psnr_val,
            'rmse': rmse_val,
            'hfen': hfen_val
        })

        scipy.io.savemat(os.path.join(exp_dir, f"patient_{i}_orientation_{j}.mat"), {"chi_recon": x_k_cpu})

#%% Save Summary Results
df = pd.DataFrame(results)
save_path = os.path.join(exp_dir, "momentum_qsm_l2_results.csv")
df.to_csv(save_path, index=False)
print(f"Results saved to {save_path}")

avg_results = df.mean(numeric_only=True)
print("\n=== Average Results ===")
print(avg_results)

mean_vals = df.mean(numeric_only=True)
std_vals = df.std(numeric_only=True)

summary = pd.DataFrame({
    'Metric': mean_vals.index,
    'Mean ± Std': [
        f"{mean_vals[m]:.4e} ± {std_vals[m]:.4e}" if m == 'loss'
        else f"{mean_vals[m]:.4f} ± {std_vals[m]:.4f}"
        for m in mean_vals.index
    ]
})
print(summary)
