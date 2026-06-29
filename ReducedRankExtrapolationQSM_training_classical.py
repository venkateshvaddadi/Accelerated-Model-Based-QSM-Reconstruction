#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 11:01:39 2026

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

#%%

# ============================================================
# 2. Simple 3D Denoiser (Replace with ISDU / U-Net)
# ============================================================


no_channels=32

class Conv_ReLU_Block(nn.Module):
	def __init__(self):
		super(Conv_ReLU_Block, self).__init__()
		self.conv = nn.Conv3d(in_channels=no_channels, out_channels=no_channels, kernel_size=3, stride=1, padding=1, bias=False)
		self.relu = nn.ReLU(inplace=True)
		
	def forward(self, x):
		return self.relu(self.conv(x))

class BasicBlock(nn.Module):

    def __init__(self, inplanes=no_channels, planes=no_channels):
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

	def __init__(self, inplanes=no_channels, planes=no_channels, dropout_rate=0.5):
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
				nn.Conv3d(in_channels=1, out_channels=no_channels, kernel_size=3, stride=1, padding=1, bias=False),
				nn.ReLU(inplace=True),
				self.make_layer(wBasicBlock, 1),
				nn.Conv3d(in_channels=no_channels, out_channels=no_channels, kernel_size=1, stride=1, padding=0, bias=False),
				nn.ReLU(inplace=True),
				nn.Conv3d(in_channels=no_channels, out_channels=no_channels, kernel_size=1, stride=1, padding=0, bias=False),
				nn.ReLU(inplace=True),
				nn.Conv3d(in_channels=no_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
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

# Replace device_id = 0
device = torch.device('cpu')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on: {device}")
# Parameters for dipole kernel
matrix_size = [176, 176, 160]
voxel_size = [1, 1, 1]
dk = dipole_kernel(matrix_size, voxel_size, B0_dir=[0, 0, 1]).to(device)




#%%
import torch
import torch.nn as nn




# class ReducedRankExtrapolationQSM(nn.Module):
#     def __init__(self, refiner_network=None, num_iters=10, k_order=3, rho=0.5):
#         super().__init__()
#         self.refiner = refiner_network
#         self.num_iters = num_iters
#         self.k_order = k_order  # Number of iterates to use for extrapolation [cite: 216]
#         self.rho = rho
#         self.L = nn.Parameter(torch.tensor(1.0))

#     def forward(self, local_field, dipole_kernel):
#         x_curr = torch.zeros_like(local_field)
        
#         # We perform cycles of 'k_order' steps followed by one extrapolation 'jump' [cite: 218]
#         for i in range(self.num_iters // self.k_order):
            
#             # --- 1. Generate a sequence of k+1 iterates ---
#             sequence = [x_curr.clone()]
#             for _ in range(self.k_order):
#                 # Standard MBIR / Refiner step
#                 x_next = self.step_logic(sequence[-1], local_field, dipole_kernel)
#                 sequence.append(x_next)
            
#             # --- 2. Apply RRE Extrapolation ---
#             # Compute differences Δx_i [cite: 225]
#             U = []
#             for j in range(len(sequence) - 1):
#                 U.append((sequence[j+1] - sequence[j]).view(-1))
            
#             # U matrix of differences 
#             U = torch.stack(U).T 
            
#             # Solve for coefficients γ using the RRE logic [cite: 213, 232]
#             # Solving R* R d = 1 (Simplified RRE)
#             Q, R = torch.linalg.qr(U)
#             ones = torch.ones(self.k_order, 1, device=x_curr.device)
#             # R* R d = 1 -> R d = Q* 1 (least squares)
#             d = torch.linalg.lstsq(R, torch.linalg.lstsq(R.T, ones).solution).solution
#             gamma = d / torch.sum(d) # Normalize [cite: 233]
            
#             # --- 3. Compute Extrapolated Limit ---
#             x_extrapolated = torch.zeros_like(x_curr)
#             for j in range(self.k_order):
#                 x_extrapolated += gamma[j] * sequence[j]
            
#             x_curr = x_extrapolated # Restart cycle from the jump point [cite: 219]

#         return x_curr

#     def step_logic(self, x, y, dk):
#         # Your core QSM physics + Refiner update
#         if self.refiner is not None:
#             z = (1 - self.rho) * x + self.rho * self.refiner(x)
#         else:
#             z = x
            
#         residual = self.dipole_conv(z, dk) - y
#         grad = self.dipole_conv(residual, dk)
#         return z - (1.0 / self.L) * grad

#     def dipole_conv(self, x, kernel):
#         return torch.real(torch.fft.ifftn(torch.fft.fftn(x) * kernel))

# import torch
# import torch.nn as nn

# class ReducedRankExtrapolationQSM(nn.Module):
#     def __init__(self, refiner_network=None, num_iters=12, k_order=3, rho=0.5):
#         super().__init__()
#         self.refiner = refiner_network
#         self.num_iters = num_iters
#         self.k_order = k_order  # Number of iterates used for extrapolation [cite: 216]
#         self.rho = nn.Parameter(torch.tensor(rho))

#         # L: Majorizer/step-size parameter [cite: 191]
#         self.L = nn.Parameter(torch.tensor(1.0))

#     def forward(self, local_field, dipole_kernel):
        
#         print(self.num_iters,self.k_order,self.rho,self.L)
        
#         x_curr = torch.zeros_like(local_field)
        
#         # We perform cycles of 'k_order' steps followed by one extrapolation 'jump' [cite: 144, 218]
#         # Total iterations must be divisible by k_order for full utilization.
#         for i in range(self.num_iters // self.k_order):
            
#             # --- 1. Generate a sequence of k+1 iterates ---
#             sequence = [x_curr.clone()]
#             for _ in range(self.k_order):
#                 # Apply standard MBIR / Refiner step logic
#                 x_next = self.step_logic(sequence[-1], local_field, dipole_kernel)
#                 sequence.append(x_next)
            
#             # --- 2. Apply RRE Extrapolation ---
#             # Compute differences Δx_i 
#             U = []
#             for j in range(len(sequence) - 1):
#                 # FIX: Use .reshape(-1) to avoid RuntimeError with non-contiguous tensors
#                 delta = (sequence[j+1] - sequence[j]).reshape(-1)
#                 U.append(delta)
            
#             # Stack to create the matrix U [cite: 211, 226]
#             # Matrix U has dimensions (N x k_order)
#             U = torch.stack(U).T 
            
#             # Solve for coefficients γ using the RRE logic [cite: 213, 232]
#             # We solve R* R d = 1 via QR decomposition for numerical stability [cite: 227]
#             Q, R = torch.linalg.qr(U)
#             ones = torch.ones(self.k_order, 1, device=x_curr.device)
            
#             # Use least squares to find the solution d [cite: 203]
#             # d is the weights for the minimal polynomial logic
#             d = torch.linalg.lstsq(R, torch.linalg.lstsq(R.T, ones).solution).solution
#             gamma = d / torch.sum(d) # Normalize weights [cite: 180, 233]
            
#             # --- 3. Compute Extrapolated Limit ---
#             # The extrapolated estimate s is a weighted sum of the iterates [cite: 185, 213]
#             x_extrapolated = torch.zeros_like(x_curr)
#             for j in range(self.k_order):
#                 x_extrapolated += gamma[j] * sequence[j]
            
#             # Restart the next cycle from the extrapolated jump point [cite: 219]
#             x_curr = x_extrapolated 

#         return x_curr

#     def step_logic(self, x, y, dk):
#         """Standard iterative refinement step for QSM physics."""
#         # Optional Image Refining Module [cite: 165]
#         if self.refiner is not None:
#             z = (1 - self.rho) * x + self.rho * self.refiner(x)
#         else:
#             z = x
            
#         # MBIR Module: Gradient descent on the physical dipole model [cite: 37, 187]
#         # grad = D^T * (D*x - y)
#         residual = self.dipole_conv(z, dk) - y
#         grad = self.dipole_conv(residual, dk) # D is self-adjoint
        
#         return z - (1.0 / self.L) * grad

#     def dipole_conv(self, x, kernel):
#         """QSM physics performed in Fourier Domain [cite: 193]"""
#         return torch.real(torch.fft.ifftn(torch.fft.fftn(x) * kernel))


import torch
import torch.nn as nn


class ReducedRankExtrapolationQSM(nn.Module):
    """
    Reduced Rank Extrapolation (RRE) accelerated QSM reconstruction.

    Based on:
    Awasthi et al., JBO 2018 (Vector Extrapolation Methods)

    Adapted for QSM physics: D^T (D x - y)
    """

    def __init__(self, refiner_network=None, num_iters=12, k_order=3, rho=0.5):
        super().__init__()

        self.refiner = refiner_network
        self.num_iters = num_iters
        self.k_order = k_order  # Number of inner iterations (k)
        
        # Learnable parameters
        self.rho = nn.Parameter(torch.tensor(rho, dtype=torch.float32))
        self.L = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, local_field, dipole_kernel):
        """
        Args:
            local_field: measured field (y)
            dipole_kernel: dipole kernel in Fourier domain (D)

        Returns:
            susceptibility map (x)
        """

        x_curr = torch.zeros_like(local_field)

        num_cycles = self.num_iters // self.k_order

        for _ in range(num_cycles):

            # ---------------------------------------------------
            # 1. Generate k+1 iterates: x0, x1, ..., xk
            # ---------------------------------------------------
            sequence = [x_curr]

            for _ in range(self.k_order):
                x_next = self.step_logic(sequence[-1], local_field, dipole_kernel)
                sequence.append(x_next)

            # ---------------------------------------------------
            # 2. Construct difference matrix U
            #    U = [x1-x0, x2-x1, ..., xk - x(k-1)]
            # ---------------------------------------------------
            deltas = [
                (sequence[i + 1] - sequence[i]).reshape(-1)
                for i in range(self.k_order)
            ]

            U = torch.stack(deltas, dim=1)  # Shape: [N, k]

            # ---------------------------------------------------
            # 3. Solve RRE system: (R^T R) d = 1
            # ---------------------------------------------------
            try:
                Q, R = torch.linalg.qr(U, mode='reduced')

                ones = torch.ones(self.k_order, 1, device=U.device)

                # Solve R^T y = 1
                y = torch.linalg.lstsq(R.T, ones).solution

                # Solve R d = y
                d = torch.linalg.lstsq(R, y).solution

            except RuntimeError:
                # Fallback (numerical instability)
                UTU = U.T @ U + 1e-6 * torch.eye(self.k_order, device=U.device)
                ones = torch.ones(self.k_order, 1, device=U.device)
                d = torch.linalg.solve(UTU, ones)

            # Normalize weights
            gamma = d / torch.sum(d)

            # ---------------------------------------------------
            # 4. Extrapolated solution (IMPORTANT FIX)
            #    Use k+1 iterates
            # ---------------------------------------------------
            x_extrapolated = torch.zeros_like(x_curr)

            for j in range(self.k_order + 1):
                weight = gamma[j] if j < self.k_order else (1 - torch.sum(gamma))
                x_extrapolated += weight * sequence[j]

            # Restart
            x_curr = x_extrapolated

        return x_curr


    def step_logic(self, x, y, dk):
        """
        One iteration of model-based QSM update:
        x_{k+1} = z - (1/L) * D^T (D z - y)
        """

        # -----------------------------
        # Learned refinement (denoiser)
        # -----------------------------
        if self.refiner is not None:
            z = (1 - self.rho) * x + self.rho * self.refiner(x)
        else:
            z = x

        # -----------------------------
        # Physics-based gradient
        # -----------------------------
        residual = self.dipole_conv(z, dk) - y
        grad = self.dipole_conv(residual, dk)  # D^T = D (self-adjoint)

        # -----------------------------
        # Gradient descent update
        # -----------------------------
        return z - (1.0 / self.L) * grad


    def dipole_conv(self, x, kernel):
        """
        Dipole convolution in Fourier domain:
        D(x) = F^{-1}( F(x) * kernel )
        """
        return torch.real(
            torch.fft.ifftn(torch.fft.fftn(x) * kernel)
        )


















#%%




import os
from datetime import datetime

# ==========================================
# Experiment Setup
# ==========================================

base_dir = "savedModels/ReducedRankExtrapolationQSM"

# Add meaningful naming (VERY useful later 🔥)
model_name = "ReducedRankExtrapolationQSM"
device_id = 0
batch_size = 2
num_iters = 6
k_order = 3  # Paper uses low-order extrapolation (e.g., quadratic) [cite: 425]
rho = 0.5


# Replace device_id = 0
# device = torch.device('cpu')
# %%
# Create unique experiment name

timestamp = datetime.now().strftime("%b_%d_%H_%M_%S")
exp_name = f"RRE_QSM_K_{num_iters}_order_{k_order}_{timestamp}"
exp_name = f"RRE_QSM_K_{num_iters}_order_{k_order}"

# Full path
exp_dir = os.path.join(base_dir, exp_name)

# Create folders
os.makedirs(exp_dir, exist_ok=True)

#%%
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
print(f"💻 Device:         device")
print("="*50 + "\n")



#%%
wide_resnet_model = WideResNet().cuda(device_id)
# # model = MomentumNetQSM(num_iters=num_iters, rho=rho).to(device_id)


# encoder_norm = nn.Identity
# norm = nn.InstanceNorm3d
# use_skip = False

# from modules.model_from_DIP import DIPNet

 # refinment_model = DIPNet(depth=1, base=16, decoder_block_num=1, encoder_norm=encoder_norm, norm=norm, use_skip=use_skip)


model = ReducedRankExtrapolationQSM(refiner_network=None, num_iters=num_iters, k_order=k_order, rho=rho).to(device)

#%%

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
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
#%%



def sobel_kernel():
    s = [
        [
            [1,   2,   1],
            [2,   4,   2],
            [1,   2,   1]
        ],
        [
            [0,   0,   0],
            [0,   0,   0],
            [0,   0,   0]
        ],
        [
            [-1, -2, -1],
            [-2, -4, -2],
            [-1, -2, -1]
        ]
    ]
    s = torch.FloatTensor(s)
    sx = s
    sy = s.permute(1, 2, 0)
    sz = s.permute(2, 0, 1)
    ss = torch.stack([sx, sy, sz]).unsqueeze(1)

    return ss


ss = sobel_kernel()
ss=ss.float()
ss = ss.to(device)


def total_loss_l1(chi, y, b, d, m, sobel):    
    
    # chi = predicetd susc
    # y   = cosmos susc
    # b   = phs
    # d   = dipole kernel
    # m   = mask
    # y_mean = label mean
    # y_std  = label std
    # b_mean = input_mean
    # b_std  = input_std
    
    def _l1error(x1, x2):
        return torch.mean(torch.abs(x1 - x2))


    def _chi_to_b(chi, b, d, m):
        
        # chi = predicetd susc
        # b   = phs
        # d   = dipole kernel
        # m   = mask
        
        chi_fourier = torch.fft.fftn(chi,dim=[2,3,4])
        b_hat_fourier = (chi_fourier * d)
        b_hat = torch.real(torch.fft.ifftn(b_hat_fourier,dim=[2,3,4]))
    
        # Multiply masks
        b = b * m
        b_hat = b_hat * m    
        return b, b_hat
    
    def loss_l1(chi, y):
        return _l1error(chi, y)


    def loss_model(b, b_hat):    
        return _l1error(b, b_hat)
    
    def loss_gradient(chi, y, sobel):
        temp=y - chi
        difference   = F.conv3d(temp.float()  ,   sobel.float(), padding=1)
        return torch.mean(torch.abs(difference))
    
    print("we are using model loss with l1 norm")
    b, b_hat = _chi_to_b(chi, b, d, m)
    
    loss_model = loss_model(b, b_hat)
    loss_grad  = loss_gradient(b, b_hat, sobel)
    loss = 0.9*loss_model+ 0.1* loss_grad
    
    return loss


#%%3







        
#%%
batch_size=1
results = []

patients_list = [1,2,3,4,5,6,7,8,9,10, 11,12] 
# patients_list = [7,8,9,10, 11,12] 

orientations=[1,2,3,4,5]

patients_list = [1 ] 
orientations=[ 1]

raw_data_path = '../../QSM_data/data_for_experiments/given_data/raw_data_names_modified/'

for itetations in range(100):

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
    
            phs = torch.tensor(phs).float().unsqueeze(0).unsqueeze(0).to(device)
            msk = torch.tensor(msk).float().unsqueeze(0).unsqueeze(0).to(device)
            mag = torch.tensor(mag).float().unsqueeze(0).unsqueeze(0).to(device)
            sus = torch.tensor(sus).float().unsqueeze(0).unsqueeze(0).to(device)
    
            dk_rep = dk.repeat(batch_size, 1, 1, 1, 1)
    
    
            for epoch in range(1):        
    
                tic()
                chi_pred = model(phs, dk_rep)
                toc()
                phi_pred = torch.fft.ifftn(dk_rep * torch.fft.fftn(chi_pred, dim=[2,3,4]), dim=[2,3,4])
                
        
                # ----------------------------------
                # Loss
                # ----------------------------------
                # loss = criterion(phi_pred.real, phs)
                loss = 0.9*criterion(phi_pred.real, phs) + 0.1 * gradient_loss(chi_pred)
                # loss = torch.mean(mag * (phi_pred.real - phs)**2)
                #loss=total_loss_l1(chi=chi_pred.real, y=sus, b=phs, d=dk, m=msk, sobel=ss)
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
                sus=scipy.io.loadmat(raw_data_path+'/patient_'+str(i)+'/cos'+str(j)+'.mat')['cos']
    
                psnr_val = compute_psnr(x_k_cpu, sus)
                rmse_val = compute_rmse(x_k_cpu, sus)
                hfen_val = compute_hfen(x_k_cpu, sus)
    
                # print(sus.shape)            
                sus = np.expand_dims(sus, axis=0)  # Shape becomes (1, 3)
                sus = np.expand_dims(sus, axis=0)  # Shape becomes (1, 3)
    
                # print(sus.shape)
    
    
                ssim_val=compute_ssim_numpy(x_k_cpu, sus)
                
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

#%%


df = pd.DataFrame(results)
save_path = exp_dir+"/qsm_results.csv"
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
















