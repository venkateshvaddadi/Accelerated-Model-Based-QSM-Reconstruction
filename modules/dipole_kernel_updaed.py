#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 20:56:39 2021

@author: venkatesh
"""


import numpy as np

import torch

#%%
import numpy as np
import torch

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

matrix_size = [64, 64, 64]
voxel_size = [1,  1,  1]
B0_dir=[0,0,1]
D=dipole_kernel(matrix_size,voxel_size,B0_dir)
