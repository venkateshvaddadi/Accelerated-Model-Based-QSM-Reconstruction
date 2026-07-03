#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 12:26:05 2026

@author: venkatesh
"""

import os
import scipy.io
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn

# Internal Imports
from models import ReducedRankExtrapolationQSM
from utils import dipole_kernel, sobel_kernel, gradient_loss, tic, toc, plot_qsm_comparison
from modules.metrics import compute_psnr, compute_rmse, compute_hfen, compute_ssim_numpy

# --- Runtime Configuration & Device ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_dir = "savedModels/ReducedRankExtrapolationQSM"
model_name = "ReducedRankExtrapolationQSM"
matrix_size = [176, 176, 160]
voxel_size = [1, 1, 1]

batch_size = 1
num_iters = 6
k_order = 3
rho = 0.5
lr = 1e-4

# Build Experiment Dir
timestamp = datetime.now().strftime("%b_%d_%H_%M_%S")
exp_name = f"RRE_QSM_K_{num_iters}_order_{k_order}_{timestamp}"
exp_dir = os.path.join(base_dir, exp_name)
os.makedirs(exp_dir, exist_ok=True)

print("\n" + "="*50)
print(f"🚀 STARTING EXPERIMENT: {model_name}")
print("="*50)
print(f"📂 Save Directory:  {exp_dir}")
print(f"💻 Device:          {device}")
print("="*50 + "\n")

# --- Initialize Modules ---
dk = dipole_kernel(matrix_size, voxel_size).to(device)
ss = sobel_kernel().float().to(device)

model = ReducedRankExtrapolationQSM(refiner_network=None, num_iters=num_iters, k_order=k_order, rho=rho).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# --- Patient Evaluation Loop ---
results = []
patients_list = list(range(1, 13)) 
orientations = list(range(1, 6))
raw_data_path = '../../QSM_data/data_for_experiments/given_data/raw_data_names_modified/'

for epoch in range(1):
    for idx in patients_list:
        total_loss = 0
        for o in orientations:
            # Data Paths
            p_path = f"{raw_data_path}/patient_{idx}"
            phs_raw = scipy.io.loadmat(f"{p_path}/phs{o}.mat")['phs']
            msk_raw = scipy.io.loadmat(f"{p_path}/msk{o}.mat")['msk']
            mag_raw = scipy.io.loadmat(f"{p_path}/mag{o}.mat")['mag']
            sus_raw = scipy.io.loadmat(f"{p_path}/cos{o}.mat")['cos']
            
            # Tensors Setup
            phs = torch.tensor(phs_raw).float().unsqueeze(0).unsqueeze(0).to(device)
            msk = torch.tensor(msk_raw).float().unsqueeze(0).unsqueeze(0).to(device)
            mag = torch.tensor(mag_raw).float().unsqueeze(0).unsqueeze(0).to(device)
            sus = torch.tensor(sus_raw).float().unsqueeze(0).unsqueeze(0).to(device)
            
            dk_rep = dk.repeat(batch_size, 1, 1, 1, 1)
            
            # Train step
            tic()
            chi_pred = model(phs, dk_rep)
            toc()
            
            phi_pred = torch.fft.ifftn(dk_rep * torch.fft.fftn(chi_pred, dim=[2,3,4]), dim=[2,3,4])
            loss = 0.9 * criterion(phi_pred.real, phs) + 0.1 * gradient_loss(chi_pred)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            x_k_cpu = (chi_pred.real.detach().cpu().numpy()) * (msk.detach().cpu().numpy())
            
            # Metric Evaluations
            psnr_val = compute_psnr(x_k_cpu, sus_raw)
            rmse_val = compute_rmse(x_k_cpu, sus_raw)
            hfen_val = compute_hfen(x_k_cpu, sus_raw)
            
            sus_expanded = np.expand_dims(np.expand_dims(sus_raw, axis=0), axis=0)
            ssim_val = compute_ssim_numpy(x_k_cpu, sus_expanded)
            
            print({
                'patient': idx, 'orientation': o,
                'SSIM': f'{ssim_val:.4f}', 'Loss': f'{loss.item():.2e}',
                'PSNR': f'{psnr_val:.2f}', 'RMSE': f'{rmse_val:.4f}', 'HFEN': f'{hfen_val:.4f}'
            })
            
            results.append({
                'patient': idx, 'orientation': o, 'loss': loss.item(),
                'ssim': ssim_val.item(), 'psnr': psnr_val, 'rmse': rmse_val, 'hfen': hfen_val
            })
            
            # Save Predictions
            scipy.io.savemat(os.path.join(exp_dir, f"patient_{idx}_orientation_{o}.mat"), {"chi_recon": x_k_cpu})

# --- Save Dataframes and Print Summary Logs ---
df = pd.DataFrame(results)
save_path = os.path.join(exp_dir, "qsm_results.csv")
df.to_csv(save_path, index=False)
print(f"\nSaved CSV tracking output to: {save_path}")

mean_vals = df.mean(numeric_only=True)
std_vals = df.std(numeric_only=True)

summary = pd.DataFrame({
    'Metric': mean_vals.index,
    'Mean ± Std': [
        f"{mean_vals[m]:.2e} ± {std_vals[m]:.2e}" if m == 'loss'
        else f"{mean_vals[m]:.2f} ± {std_vals[m]:.2f}" for m in mean_vals.index
    ]
})
print("\n=== Final Quantitative Summary ===")
print(summary)

# Visual Verification
plot_qsm_comparison(x_k_cpu, sus_raw, msk_raw, title=f"Patient {idx} Orientation {o}")