#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Momentum-Accelerated Model-Based QSM (MA-Mo-QSM)

Official implementation accompanying the manuscript:
"Accelerated Model-Based Quantitative Susceptibility Mapping
Using Momentum and Vector Extrapolation"

Author:
Venkatesh Vaddadi
Indian Institute of Science (IISc)
"""

# --- Standard Library & OS ---
# ==========================================================
# Standard Library
# ==========================================================
import os
import time
from datetime import datetime

# ==========================================================
# Scientific Computing
# ==========================================================
import numpy as np
import pandas as pd
import scipy.io

# ==========================================================
# PyTorch
# ==========================================================
import torch
import torch.nn as nn
import torch.fft

# ==========================================================
# Visualization
# ==========================================================
import matplotlib.pyplot as plt

def dipole_kernel(matrix_size, voxel_size, B0_dir=[0, 0, 1]):
    # Fix: Replace np.int with built-in int for NumPy 2.0+ compatibility
    # matrix_size is expected to be [depth, height, width]
    [Y, X, Z] = np.meshgrid(
        np.linspace(-int(matrix_size[1]/2), int(matrix_size[1]/2)-1, matrix_size[1]),
        np.linspace(-int(matrix_size[0]/2), int(matrix_size[0]/2)-1, matrix_size[0]),
        np.linspace(-int(matrix_size[2]/2), int(matrix_size[2]/2)-1, matrix_size[2])
    )
    
    X = X / (matrix_size[0]) * voxel_size[0]
    Y = Y / (matrix_size[1]) * voxel_size[1]
    Z = Z / (matrix_size[2]) * voxel_size[2]
    
    # Calculate Dipole Kernel: D = 1/3 - (kz^2 / k^2)
    numerator = np.square(X * B0_dir[0] + Y * B0_dir[1] + Z * B0_dir[2])
    denominator = np.square(X) + np.square(Y) + np.square(Z) + np.finfo(float).eps
    D = 1/3 - np.divide(numerator, denominator)
    
    D = np.where(np.isnan(D), 0, D)

    # Fix: Replace np.int with int for roll operations
    D = np.roll(D, int(np.floor(matrix_size[0]/2)), axis=0)
    D = np.roll(D, int(np.floor(matrix_size[1]/2)), axis=1)
    D = np.roll(D, int(np.floor(matrix_size[2]/2)), axis=2)
    
    D = np.float32(D)
    # Return as a torch tensor with a batch dimension for your DE-PROX model
    D = torch.tensor(D).unsqueeze(dim=0)
    
    return D

# ==========================================================

device_id = 0

# Parameters for dipole kernel
matrix_size = [176, 176, 160]
voxel_size = [1, 1, 1]
dk = dipole_kernel(matrix_size, voxel_size, B0_dir=[0, 0, 1]).cuda(device_id)

# ==========================================================

class MomentumNetQSM(nn.Module):
    def __init__(self, num_iters=10, rho=0.5):
        super().__init__()
        self.num_iters = num_iters
        self.rho = rho # Relaxation parameter [cite: 163, 165]
        self.L = nn.Parameter(torch.tensor(1.0))        

    def forward(self, local_field, dipole_kernel):
        # x: Susceptibility map (chi), y: Local field
        x_curr = torch.zeros_like(local_field)
        x_prev = torch.zeros_like(local_field)
        
        # Initialization of momentum parameters [cite: 202, 203]
        theta_prev = 1.0
        
        for i in range(self.num_iters):
            # --- 1. Image Refining Module ---
            # Output z is a mix of previous estimate and network refinement [cite: 165]
            # z = (1 - self.rho) * x_curr + self.rho * self.refiner(x_curr)
            
            # --- 2. Extrapolation Module (Momentum) ---
            # Calculate momentum coefficient m [cite: 202]
            theta_curr = (1 + (1 + 4 * theta_prev**2)**0.5) / 2
            m = (theta_prev - 1) / theta_curr
            
            # Extrapolated point [cite: 169]
            x_hat = x_curr + m * (x_curr - x_prev)
            
            # --- 3. MBIR Module (QSM Physics) ---
            # This is the proximal gradient step for the dipole kernel
            # grad = D^T * (D*x_hat - local_field)
            residual = self.dipole_conv(x_hat, dipole_kernel) - local_field
            grad = self.dipole_conv(residual, dipole_kernel) # D is self-adjoint
            
            x_next = x_hat - (1.0 / self.L) * grad # L is the majorizer/step-size
            
            # Update for next iteration
            x_prev = x_curr
            x_curr = x_next
            theta_prev = theta_curr
            
        return x_curr

    def dipole_conv(self, x, kernel):
        # QSM physics performed in Fourier Domain
        return torch.real(torch.fft.ifftn(torch.fft.fftn(x) * kernel))

# ==========================================================

import os
from datetime import datetime

# ==========================================
# Experiment Setup
# ==========================================

base_dir = "savedModels/MOMENTUM_QSM_EXPERIMENTS/"

# Add meaningful naming (VERY useful later 🔥)
model_name = "MomentumQSM"
num_iters = 9
rho = 0.5

# %%
# Create unique experiment name

timestamp = datetime.now().strftime("%b_%d_%H_%M_%S")
exp_name = f"{model_name}_K_{num_iters}"

# Full path
exp_dir = os.path.join(base_dir, exp_name)

# Create folders
os.makedirs(exp_dir, exist_ok=True)

# ==========================================================

# ==========================================
# Display Experiment Metadata
# ==========================================
print("\n" + "="*50)
print(f"🚀 STARTING EXPERIMENT: {model_name}")
print("="*50)
print(f"📂 Save Directory:  {exp_dir}")
print(f"🕒 Timestamp:       {timestamp}")
print(f"🔄 Iterations (K):  {num_iters}")
print(f"🧬 Relaxation (ρ):  {rho}")
print(f"💻 Device:          CUDA:{device_id}")
print("="*50 + "\n")

# ==========================================================

model = MomentumNetQSM(num_iters=num_iters, rho=rho).to(device_id)

# ==========================================================

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# ==========================================================

def gradient_loss(gen_chi):
    # Penalizes sharp, unphysical jumps in the susceptibility map
    dx = gen_chi[:, :, 1:, :, :] - gen_chi[:, :, :-1, :, :]
    dy = gen_chi[:, :, :, 1:, :] - gen_chi[:, :, :, :-1, :]
    dz = gen_chi[:, :, :, :, 1:] - gen_chi[:, :, :, :, :-1]
    return torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy)) + torch.mean(torch.abs(dz))

# Total Loss

# ==========================================================

def tic():
    # Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        #print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
        print(str(time.time() - startTime_for_tictoc) )
    else:
        print("Toc: start time not set")

# ==========================================================

batch_size=1
results = []

patients_list = [1,2,3,4,5,6,7,8,9,10,11,12] 
orientations=[1,2,3,4,5]

raw_data_path = '../../QSM_data/data_for_experiments/given_data/raw_data_names_modified/'

for i in patients_list:
    # print(f"Patient: {i}\n")
    total_loss=0
    for j in orientations:
        # print(f'Orientation: {j}')

        # Load Raw Data
        phs = scipy.io.loadmat(f"{raw_data_path}/patient_{i}/phs{j}.mat")['phs']
        msk = scipy.io.loadmat(f"{raw_data_path}/patient_{i}/msk{j}.mat")['msk']

        sus=scipy.io.loadmat(raw_data_path+'/patient_'+str(i)+'/cos'+str(j)+'.mat')['cos']

        phs = torch.tensor(phs).float().unsqueeze(0).unsqueeze(0).cuda(device_id)
        msk = torch.tensor(msk).float().unsqueeze(0).unsqueeze(0).cuda(device_id)

        temp_shape=phs.shape
        matrix_size = [temp_shape[2],temp_shape[3], temp_shape[4]]
        voxel_size = [1,  1,  1]
        dk = dipole_kernel(matrix_size, voxel_size, B0_dir=[0, 0, 1]).cuda(device_id)
        dk_rep = dk.repeat(batch_size, 1, 1, 1, 1)

        for epoch in range(1):        

            tic()
            chi_pred = model(phs, dk_rep)
            toc()

            phi_pred = torch.fft.ifftn(dk_rep * torch.fft.fftn(chi_pred, dim=[2,3,4]), dim=[2,3,4])
    
    
            # ----------------------------------
            # Loss
            # ----------------------------------
            loss = torch.mean((phi_pred.real - phs)**2)

            # ----------------------------------
            # Backprop
            # ----------------------------------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item()

        
            x_k_cpu = (chi_pred.real.detach().cpu().numpy()) * (msk.detach().cpu().numpy())

            from modules.metrics import *
            # =========================
            # COMPUTE METRICS (NEW)
            # =========================
            psnr_val = compute_psnr(x_k_cpu, sus)
            rmse_val = compute_rmse(x_k_cpu, sus)
            hfen_val = compute_hfen(x_k_cpu, sus)

            sus_copy=np.copy(sus)            

            # print(sus.shape)            
            sus_copy = np.expand_dims(sus_copy, axis=0)  # Shape becomes (1, 3)
            sus_copy = np.expand_dims(sus_copy, axis=0)  # Shape becomes (1, 3)

            # print(sus.shape)

            ssim_val=compute_ssim_numpy(x_k_cpu, sus_copy)
            
            # 🔥 UPDATE PROGRESS BAR INSTEAD OF PRINTING
            print({
                'patient':i,
                'orientation':j,
                'SSIM': f'{ssim_val:.4f}',
                'Loss': f'{loss.item():.2e}',
                'PSNR': f'{psnr_val:.2f}',
                'RMSE': f'{rmse_val:.4f}',
                'HFEN': f'{hfen_val:.4f}',
            })

            result_dict = {
                            'patient': i,
                            'orientation': j,
                            'loss': loss.item(),
                            'ssim':ssim_val.item(),
                            'psnr': psnr_val,
                            'rmse': rmse_val,
                            'hfen': hfen_val
                        }
                        
            results.append(result_dict)
            
            # Optional: still print
            # print({k: (f"{v:.4e}" if k=='loss' else f"{v:.4f}") for k, v in result_dict.items()})

            import scipy.io

            recon_mat_path = os.path.join(
                exp_dir, f"patient_{i}_orientation_{j}.mat"
            )
            
            scipy.io.savemat(recon_mat_path, {
                "chi_recon": x_k_cpu
            })

df = pd.DataFrame(results)
save_path = "momentum_qsm_results.csv"
df.to_csv(save_path, index=False)

print(f"Results saved to {save_path}")

# ==========================================================

avg_results = df.mean(numeric_only=True)

print("\n=== Average Results ===")
print(avg_results)

# ==========================================================

df = pd.DataFrame(results)

mean_vals = df.mean(numeric_only=True)
std_vals = df.std(numeric_only=True)

# Combine into "mean ± std" format
summary = pd.DataFrame({
    'Metric': mean_vals.index,
    'Mean ± Std': [
        f"{mean_vals[m]:.2e} ± {std_vals[m]:.2e}" if m == 'loss'
        else f"{mean_vals[m]:.2f} ± {std_vals[m]:.2f}"
        for m in mean_vals.index
    ]
})

print(summary)

# ==========================================================

import matplotlib.pyplot as plt

def plot_qsm_comparison(reconstruction, ground_truth, mask, title="QSM Reconstruction"):
    # 1. Remove batch/channel dims and apply mask
    # Shape logic: [1, 1, 176, 176, 160] -> [176, 176, 160]
    recon = np.squeeze(reconstruction) * np.squeeze(mask)
    gt = np.squeeze(ground_truth) * np.squeeze(mask)
    
    # 2. Select the middle slice along the axial direction (Z-axis)
    mid_slice = recon.shape[2] // 2
    recon_slice = recon[:, :, mid_slice]
    gt_slice = gt[:, :, mid_slice]
    error_slice = np.abs(gt_slice - recon_slice)

    # 3. Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Determine color scale (clipping outliers for better contrast)
    vmin, vmax = -0.1, 0.1 

    im1 = axes[0].imshow(gt_slice, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0].set_title("Ground Truth (COSMOS)")
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(recon_slice, cmap='gray', vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Reconstruction")
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    im3 = axes[2].imshow(error_slice, cmap='hot', vmin=0, vmax=0.05)
    axes[2].set_title("Absolute Error Map")
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# --- Call this inside your loop or after training ---
# plot_qsm_comparison(x_k_cpu, chi_true, msk.cpu().numpy())
# Plot the result
plot_qsm_comparison(x_k_cpu, sus, msk.detach().cpu().numpy(), 
                    title=f"Patient {i} Orientation {j}")
