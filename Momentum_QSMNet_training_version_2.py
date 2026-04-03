#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 22:25:39 2026

@author: venkatesh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 17:30:12 2026

@author: venkatesh
"""
# --- Standard Library & OS ---
import os
import time
from datetime import datetime
from pathlib import Path
from math import sqrt

# --- Scientific Computing ---
import numpy as np
import pandas as pd
import scipy.io
from tqdm import tqdm

# --- PyTorch Core ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.fft

# --- PyTorch Data & Vision ---
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

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
#%%

# ============================================================
# 2. Simple 3D Denoiser (Replace with ISDU / U-Net)
# ============================================================



class Conv_ReLU_Block(nn.Module):
	def __init__(self):
		super(Conv_ReLU_Block, self).__init__()
		self.conv = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
		self.relu = nn.ReLU(inplace=True)
		
	def forward(self, x):
		return self.relu(self.conv(x))

class BasicBlock(nn.Module):

    def __init__(self, inplanes=32, planes=32):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

class wBasicBlock(nn.Module):

	def __init__(self, inplanes=32, planes=32, dropout_rate=0.5):
		super(wBasicBlock, self).__init__()
		self.conv1 = nn.Conv3d(inplanes, planes, 3, 1, 1, bias=False)
		self.bn1 = nn.BatchNorm3d(planes)
		self.relu = nn.ReLU(inplace=True)
	
		self.dropout = nn.Dropout3d(p=dropout_rate)
	
		self.conv2 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
		self.bn2 = nn.BatchNorm3d(planes)

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.dropout(out)

		out = self.conv2(out)
		out = self.bn2(out)

		out += residual
		out = self.relu(out)

		return out

class WideResNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.alpha = nn.Parameter(torch.tensor(1.0))  # raw
		self.gen = nn.Sequential(
				nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
				nn.ReLU(inplace=True),
				self.make_layer(wBasicBlock, 1),
				nn.Conv3d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
				nn.ReLU(inplace=True),
				nn.Conv3d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
				nn.ReLU(inplace=True),
				nn.Conv3d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
		)
				
	def make_layer(self, block, num_of_layer):
		layers = []
		for _ in range(num_of_layer):
			layers.append(block())
		return nn.Sequential(*layers)

	def forward(self,x_input):
		x_pred =self.gen(x_input)
		return x_pred
    
#%%

#%%

device_id = 0

# Parameters for dipole kernel
matrix_size = [176, 176, 160]
voxel_size = [1, 1, 1]
dk = dipole_kernel(matrix_size, voxel_size, B0_dir=[0, 0, 1]).cuda(device_id)

#%%

import torch
import torch.nn as nn

class MomentumNetQSM(nn.Module):
    def __init__(self, refiner_network, num_iters=10, rho=0.7, gamma=0.001):
        super().__init__()
        self.refiner = refiner_network  # Your 3D U-Net or similar
        self.num_iters = num_iters
# %%
        self.rho =  nn.Parameter(torch.tensor(rho))  # Relaxation parameter [cite: 163, 165]

        
        # gamma: Regularization parameter balancing physics and AI refinement [cite: 37, 186]
        self.gamma = nn.Parameter(torch.tensor(gamma)) 
        
        # L: Majorizer/step-size parameter [cite: 191, 332]
        self.L = nn.Parameter(torch.tensor(1.0))        

    def forward(self, local_field, dipole_kernel):
        # x: Susceptibility map (chi), y: Local field
        x_curr = torch.zeros_like(local_field)
        x_prev = torch.zeros_like(local_field)
        
        # Initialization of momentum parameters [cite: 202, 203]
        theta_prev = 1.0
        
        for i in range(self.num_iters):
            # --- 1. Image Refining Module ---
            # Output z is a mix of previous estimate and network refinement [cite: 165, 179]
            # This 'z' is now explicitly used in the MBIR step below.
            z = (1 - self.rho) * x_curr + self.rho * self.refiner(x_curr)
            
            # --- 2. Extrapolation Module (Momentum) ---
            # Calculate momentum coefficient m [cite: 202]
            theta_curr = (1 + (1 + 4 * theta_prev**2)**0.5) / 2
            m = (theta_prev - 1) / theta_curr
            
            # Extrapolated point x_hat [cite: 169, 183]
            x_hat = x_curr + m * (x_curr - x_prev)
            
            # --- 3. MBIR Module (QSM Physics + Refinement Balance) ---
            # According to Momentum-Net, we minimize: f(x;y) + (gamma/2)||x - z||^2 [cite: 37, 187]
            
            # Part A: Gradient of the data-fit (QSM Dipole Physics)
            # grad_f = D^T * (D*x_hat - local_field)
            residual = self.dipole_conv(x_hat, dipole_kernel) - local_field
            grad_f = self.dipole_conv(residual, dipole_kernel) # D is self-adjoint
            
            # Part B: Gradient of the regularization term (Distance from refined z)
            # grad_reg = gamma * (x_hat - z)
            grad_reg = self.gamma * (x_hat - z)
            
            # Combine gradients for the proximal step 
            total_grad = grad_f + grad_reg
            x_next = x_hat - (1.0 / self.L) * total_grad
            
            # Update for next iteration
            x_prev = x_curr
            x_curr = x_next
            theta_prev = theta_curr
            
            
            print(self.rho.item())
            print(self.gamma.item())
            print(self.L.item())

        return x_curr

    def dipole_conv(self, x, kernel):
        # QSM physics performed in Fourier Domain
        # Using fftn and ifftn as the majorized MBIR step can involve unitary transforms [cite: 193]
        return torch.real(torch.fft.ifftn(torch.fft.fftn(x) * kernel))














































#%%




import os
from datetime import datetime

# ==========================================
# Experiment Setup
# ==========================================

base_dir = "savedModels/MOMENTUM_QSM_EXPERIMENTS_version_2"

# Add meaningful naming (VERY useful later 🔥)
model_name = "MomentumQSM"
num_iters = 5
rho = 0.5

# %%
# Create unique experiment name

timestamp = datetime.now().strftime("%b_%d_%H_%M_%S")
exp_name = f"{model_name}_{timestamp}_K_{num_iters}_rho_{rho}"

# Full path
exp_dir = os.path.join(base_dir, exp_name)

# Create folders
os.makedirs(exp_dir, exist_ok=True)





#%%
refinment_model = WideResNet().cuda(device_id)

encoder_norm = nn.Identity
norm = nn.InstanceNorm3d
use_skip = False

from modules.model_from_DIP import DIPNet

refinment_model = DIPNet(depth=1, base=32, decoder_block_num=1, encoder_norm=encoder_norm, norm=norm, use_skip=use_skip)


#%%

model = MomentumNetQSM(refinment_model, num_iters=num_iters, rho=rho).to(device_id)

#%%

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
#%%
def gradient_loss(gen_chi):
    # Penalizes sharp, unphysical jumps in the susceptibility map
    dx = gen_chi[:, :, 1:, :, :] - gen_chi[:, :, :-1, :, :]
    dy = gen_chi[:, :, :, 1:, :] - gen_chi[:, :, :, :-1, :]
    dz = gen_chi[:, :, :, :, 1:] - gen_chi[:, :, :, :, :-1]
    return torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy)) + torch.mean(torch.abs(dz))

# Total Loss

#%%
batch_size=1
results = []

patients_list = [1,2,3,4,5,6,7,8,9,10, 11,12] 
orientations=[1,2,3,4,5]

# patients_list = [3 ] 
# orientations=[ 5]

raw_data_path = '../../QSM_data/data_for_experiments/given_data/raw_data_names_modified/'

for i in patients_list:
    # print(f"Patient: {i}\n")
    total_loss=0
    for j in orientations:
        # print(f'Orientation: {j}')

        # Load Raw Data
        phs = scipy.io.loadmat(f"{raw_data_path}/patient_{i}/phs{j}.mat")['phs']
        msk = scipy.io.loadmat(f"{raw_data_path}/patient_{i}/msk{j}.mat")['msk']
        mag = scipy.io.loadmat(f"{raw_data_path}/patient_{i}/mag{j}.mat")['mag']

        sus=scipy.io.loadmat(raw_data_path+'/patient_'+str(i)+'/cos'+str(j)+'.mat')['cos']

        phs = torch.tensor(phs).float().unsqueeze(0).unsqueeze(0).cuda(device_id)
        msk = torch.tensor(msk).float().unsqueeze(0).unsqueeze(0).cuda(device_id)
        mag = torch.tensor(mag).float().unsqueeze(0).unsqueeze(0).cuda(device_id)

        dk_rep = dk.repeat(batch_size, 1, 1, 1, 1)

        for epoch in range(50):        


            chi_pred = model(phs, dk_rep)
    
            phi_pred = torch.fft.ifftn(dk_rep * torch.fft.fftn(chi_pred, dim=[2,3,4]), dim=[2,3,4])
    
    
            # ----------------------------------
            # Loss
            # ----------------------------------
            # loss = criterion(phi_pred.real, phs)
            # loss = criterion(phi_pred.real, phs) + 1e-3 * gradient_loss(chi_pred)
            loss = torch.mean(mag * (phi_pred.real - phs)**2)
            # ----------------------------------
            # Backprop
            # ----------------------------------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item()

            # print(chi_pred.shape)
        
        
        
            x_k_cpu = (chi_pred.real.detach().cpu().numpy()) * (msk.detach().cpu().numpy())


            from modules.metrics import *
            # =========================
            # COMPUTE METRICS (NEW)
            # =========================
            psnr_val = compute_psnr(x_k_cpu, sus)
            rmse_val = compute_rmse(x_k_cpu, sus)
            hfen_val = compute_hfen(x_k_cpu, sus)
            
            # 🔥 UPDATE PROGRESS BAR INSTEAD OF PRINTING
            print({
                'patient':i,
                'orientation':j,
                'Loss': f'{loss.item():.2e}',
                'PSNR': f'{psnr_val:.2f}',
                'RMSE': f'{rmse_val:.4f}',
                'HFEN': f'{hfen_val:.4f}',
                'epoch': f'{epoch:.2f}',

            })

            result_dict = {
                            'patient': i,
                            'orientation': j,
                            'loss': loss.item(),
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
#%%
avg_results = df.mean(numeric_only=True)

print("\n=== Average Results ===")
print(avg_results)

#%%
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

#%%
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
















