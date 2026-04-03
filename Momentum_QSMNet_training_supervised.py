#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 16:25:32 2026

@author: venkatesh
"""

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
				self.make_layer(wBasicBlock, 3),
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


#%%

import torch
import torch.nn as nn

class MomentumNetQSM(nn.Module):
    def __init__(self, refiner_network, num_iters=10, rho=0.7, gamma=0.001):
        super().__init__()
        self.refiner = refiner_network  # Your 3D U-Net or similar
        self.num_iters = num_iters
        self.rho =  nn.Parameter(torch.tensor(rho))  # Relaxation parameter [cite: 163, 165]

        # print(self.num_iters)
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
        # print(self.num_iters)
        
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
            
            
            # print(self.rho.item())
            # print(self.gamma.item())
            # print(self.L.item())

        return x_curr

    def dipole_conv(self, x, kernel):
        # QSM physics performed in Fourier Domain
        # Using fftn and ifftn as the majorized MBIR step can involve unitary transforms [cite: 193]
        return torch.real(torch.fft.ifftn(torch.fft.fftn(x) * kernel))









#%%


class mydataloader(Dataset):
    
    def __init__(self, csv_file, root_dir, training = True):
        self.names = pd.read_csv(csv_file)
        self.root_dir = root_dir  
        self.training = training
        
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        
        if self.training==True:
            file_name = os.path.join(self.root_dir,self.names['FileName'][idx])             
            data = scipy.io.loadmat(file_name)

            phs  = torch.tensor(data['phs' ]).unsqueeze(dim=0)
            msk  = torch.tensor(data['msk' ]).unsqueeze(dim=0)
            sus  = torch.tensor(data['susc']).unsqueeze(dim=0)
            
            phs=phs.float()
            sus=sus.float()
            msk=msk.float()
            
            phs=phs*msk
            sus=sus*msk

            return phs, msk, sus,file_name
        else:
            root_path=self.root_dir;
            phs_path = self.root_dir+"/phs/phs-"+str(self.names['Label'][idx])+".mat"
            msk_path = self.root_dir+"/msk/msk-"+str(self.names['Label'][idx])+".mat"
            sus_path = self.root_dir+"/cos/cos-"+str(self.names['Label'][idx])+".mat"
            file_name=self.names['FileName'][idx]
            
            phs = scipy.io.loadmat(phs_path)['phs']
            msk = scipy.io.loadmat(msk_path)['msk']
            sus = scipy.io.loadmat(sus_path)['cos']
            
            phs  = torch.tensor(phs).unsqueeze(dim=0)
            msk  = torch.tensor(msk).unsqueeze(dim=0)
            sus  = torch.tensor(sus).unsqueeze(dim=0)


             
            return phs, msk, sus,self.names['Label'][idx]



#%%%
batch_size=2
device_id = 0

csv_path = '../../QSM_data/data_for_experiments/given_data/data_source_1/'
data_path = '../../QSM_data/data_for_experiments/given_data/data_as_patches/'

csv_path='../../QSM_data/data_for_experiments/given_data/single_patient/patient_2/'
data_path='../../QSM_data/data_for_experiments/given_data/data_as_patches/'


# Setup
stats = scipy.io.loadmat(os.path.join(csv_path, 'csv_files/tr-stats.mat'))
sus_mean = torch.tensor(stats['out_mean']).cuda(device_id)
sus_std = torch.tensor(stats['out_std']).cuda(device_id)

trainloader = DataLoader(mydataloader(os.path.join(csv_path, 'csv_files/train.csv'), data_path), 
                         batch_size=batch_size, shuffle=True, drop_last=True)

valloader = DataLoader(mydataloader(os.path.join(csv_path, 'csv_files/val.csv'), data_path), 
                         batch_size=batch_size, shuffle=True, drop_last=True)






# ===== Dataset info =====
num_train_samples = len(trainloader.dataset)
num_val_samples = len(valloader.dataset)

num_train_batches = len(trainloader)
num_val_batches = len(valloader)

print("\n===== DATASET INFO =====")
print(f"Train Samples   : {num_train_samples}")
print(f"Train Batches   : {num_train_batches}")
print(f"Validation Samples : {num_val_samples}")
print(f"Validation Batches : {num_val_batches}")
print(f"Batch Size      : {batch_size}")
print("========================\n")
#%%





# Parameters for dipole kernel
matrix_size = [64,64,64]
voxel_size = [1, 1, 1]
dk = dipole_kernel(matrix_size, voxel_size, B0_dir=[0, 0, 1]).cuda(device_id)
#%%


























#%%




import os
from datetime import datetime

# ==========================================
# Experiment Setup
# ==========================================

base_dir = "savedModels/MOMENTUM_QSM_EXPERIMENTS_supervised"

# Add meaningful naming (VERY useful later 🔥)
model_name = "MomentumQSM"
num_iters = 10
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

# refinment_model = DIPNet(depth=1, base=32, decoder_block_num=1, encoder_norm=encoder_norm, norm=norm, use_skip=use_skip)


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
import tqdm
batch_size=2
results = []

patients_list = [1,2,3,4,5,6,7,8,9,10, 11,12] 
orientations=[1,2,3,4,5]

# patients_list = [3 ] 
# orientations=[ 5]

raw_data_path = '../../QSM_data/data_for_experiments/given_data/raw_data_names_modified/'

train_losses = []
val_losses = []
best_val_loss = float('inf')
history = []

# --- Training Loop ---
for epoch in range(50):
    model.train()
    training_loss=0
    validation_loss=0
    for i, (phs, msk, sus, _) in tqdm.tqdm(enumerate(trainloader)):

        phs = phs.cuda(device_id).float()
        msk = msk.cuda(device_id).float()
        sus = sus.cuda(device_id).float()

        dk_rep = dk.repeat(batch_size, 1, 1, 1, 1)

        chi_pred = model(phs, dk_rep)

        phi_pred = torch.fft.ifftn(dk_rep * torch.fft.fftn(chi_pred, dim=[2,3,4]), dim=[2,3,4])

        loss = 0.1*torch.mean( (phi_pred.real - phs)**2)+ 0.9*torch.mean( (chi_pred.real - sus)**2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        training_loss += loss.item()

        x_k_cpu = (chi_pred.real.detach().cpu().numpy()) * (msk.detach().cpu().numpy())

    model.eval()  # Switch to evaluation mode [cite: 1215, 1222]
    with torch.no_grad():  # Disable gradient tracking for efficiency and accuracy
        for i, (phs, msk, sus, _) in tqdm.tqdm(enumerate(valloader)):
    
            # print(phs.shape)
    
            phs = phs.cuda(device_id).float()
            msk = msk.cuda(device_id).float()
            sus = sus.cuda(device_id).float()
    
            dk_rep = dk.repeat(batch_size, 1, 1, 1, 1)
    
    
    
            chi_pred = model(phs, dk_rep)
    
            phi_pred = torch.fft.ifftn(dk_rep * torch.fft.fftn(chi_pred, dim=[2,3,4]), dim=[2,3,4])
    
            loss = 0.1*torch.mean( (phi_pred.real - phs)**2)+ 0.9*torch.mean( (chi_pred.real - sus)**2)
            validation_loss += loss.item()
    
            x_k_cpu = (chi_pred.real.detach().cpu().numpy()) * (msk.detach().cpu().numpy())

    training_loss /= len(trainloader)
    validation_loss /= len(valloader)
    
    train_losses.append(training_loss)
    val_losses.append(validation_loss)
    
    print(f"Epoch [{epoch}/50] | Train Loss: {training_loss:.6f} | Val Loss: {validation_loss:.6f}")


#%%
    # Save model for this epoch
    model_path = os.path.join(exp_dir, f"model_epoch_{epoch}.pth")
    
    torch.save({
        'epoch': epoch ,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': training_loss,
        'val_loss': validation_loss,
        'rho':model.rho.item(),
        'gamma': model.gamma.item(),
        'step_L':model.L.item(),
        'num_iters' :model.num_iters,


    }, model_path)
    
    print(f"💾 Saved model: {model_path}")



    if validation_loss < best_val_loss:
        best_val_loss = validation_loss
    
        best_model_path = os.path.join(exp_dir, "best_model.pth")
    
        torch.save({
            'epoch': epoch ,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': training_loss,
            'val_loss': validation_loss,
            'rho':model.rho.item(),
            'gamma': model.gamma.item(),
            'step_L':model.L.item(),
            'num_iters' :model.num_iters,

        }, best_model_path)
    
        print(f"🏆 Best model updated at epoch {epoch}")

    # At the end of each epoch, capture the current state of trainable hyperparameters
    current_rho = model.rho.item()
    current_gamma = model.gamma.item()
    current_L = model.L.item()
    
    # Update your history list or DataFrame
    history_entry = {
        'epoch': epoch ,
        'train_loss': training_loss,
        'val_loss': validation_loss,
        'rho': current_rho,
        'gamma': current_gamma,
        'step_L': current_L,
        'best_val_loss': best_val_loss
    }
    print(history_entry)
    history.append(history_entry)
    
    # Save to CSV
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(exp_dir, 'training_log.csv'), index=False)
    






















