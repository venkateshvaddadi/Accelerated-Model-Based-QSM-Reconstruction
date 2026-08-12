#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 12:12:27 2026
@author: venkatesh
"""

import numpy as np
from scipy.signal import convolve
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def compute_psnr(chi_recon, chi_true):
    """
    Compute PSNR with QSM normalization (0–255 scaling ignoring zeros)
    """
    img1 = np.asarray(chi_recon).copy()
    img2 = np.asarray(chi_true).copy()

    # QSM normalization (same as MATLAB)
    min_img = min(img1.min(), img2.min())

    img1[img1 != 0] -= min_img
    img2[img2 != 0] -= min_img

    max_img = max(img1.max(), img2.max())

    if max_img != 0:
        img1 = 255 * img1 / max_img
        img2 = 255 * img2 / max_img

    # Compute PSNR
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return np.inf  # perfect match

    PSNR = 20 * np.log10(255.0 / np.sqrt(mse))
    return PSNR

def compute_rmse(chi_recon, chi_true):
    """
    Compute normalized RMSE (percentage)
    """
    chi_recon = np.asarray(chi_recon)
    chi_true = np.asarray(chi_true)

    numerator = np.linalg.norm(chi_recon.ravel() - chi_true.ravel())
    denominator = np.linalg.norm(chi_true.ravel())

    if denominator == 0:
        return np.inf  # avoid division by zero

    rmse = 100 * numerator / denominator
    return rmse

def compute_hfen(img1, img2):
    img1 = np.squeeze(img1).astype(np.float64)
    img2 = np.squeeze(img2).astype(np.float64)
    
    if img1.ndim != 3:
        raise ValueError(f"Expected 3D volumes after squeezing, but got shape {img1.shape}")

    filt_siz = np.array([15, 15, 15])
    sig = np.array([1.5, 1.5, 1.5])
    siz = (filt_siz - 1) / 2
    
    x_range = np.arange(-siz[0], siz[0] + 1)
    y_range = np.arange(-siz[1], siz[1] + 1)
    z_range = np.arange(-siz[2], siz[2] + 1)
    x, y, z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
    
    h = np.exp(-(x**2 / (2 * sig[0]**2) + y**2 / (2 * sig[1]**2) + z**2 / (2 * sig[2]**2)))
    h = h / np.sum(h)
    
    arg = (x**2 / sig[0]**4 + y**2 / sig[1]**4 + z**2 / sig[2]**4 - 
           (1/sig[0]**2 + 1/sig[1]**2 + 1/sig[2]**2))
    
    H = arg * h
    H = H - (np.sum(H) / np.prod(filt_siz))
    
    img1_log = convolve(img1, H, mode='same')
    img2_log = convolve(img2, H, mode='same')
    
    return compute_rmse(img1_log, img2_log)

def gkernel(sigma, sk):
    """
    Exact translation of the helper function gkernel by Pierrick Coupe.
    Generates a 3D Gaussian kernel.
    """
    size_x = 2 * sk[0] + 1
    size_y = 2 * sk[1] + 1
    size_z = 2 * sk[2] + 1
    
    gaussKernel = np.zeros((size_x, size_y, size_z))
    
    for x in range(1, size_x + 1):
        for y in range(1, size_y + 1):
            for z in range(1, size_z + 1):
                radiusSquared = (x - (sk[0] + 1))**2 + (y - (sk[1] + 1))**2 + (z - (sk[2] + 1))**2
                gaussKernel[x-1, y-1, z-1] = np.exp(-radiusSquared / (2 * (sigma**2)))
                
    return gaussKernel

def compute_ssim_numpy_version_2(img1, img2, sw=None, ind=None):
    """
    Python translation of the 3D SSIM calculator for QSM.
    """
    nargin = 2
    if sw is not None:
        nargin += 1
    if ind is not None:
        nargin += 1

    if nargin < 2 or nargin > 4:
        return float('-inf'), float('-inf')

    if img1.shape != img2.shape:
        return float('-inf'), float('-inf')

    s = img1.shape
    img1 = img1.copy().astype(np.float64)
    img2 = img2.copy().astype(np.float64)

    # Make sure dynamic range is 0-255 for QSM
    min_img = min(img1.min(), img2.min())
    img1[img1 != 0] = img1[img1 != 0] - min_img
    img2[img2 != 0] = img2[img2 != 0] - min_img

    max_img = max(img1.max(), img2.max())
    img1 = 255.0 * img1 / max_img
    img2 = 255.0 * img2 / max_img

    if nargin == 2:
        sw = [2, 2, 2]
        if (s[0] < sw[0]) or (s[1] < sw[1]) or (s[2] < sw[2]):
            return float('-inf'), float('-inf')
        ind = np.where(img1 != 0)
        window = gkernel(1.5, sw)
        K = [0.01, 0.03]
        L = 255

    elif nargin == 3:
        if (s[0] < sw[0]) or (s[1] < sw[1]) or (s[2] < sw[2]):
            return float('-inf'), float('-inf')
        window = gkernel(1.5, sw)
        ind = np.where(img1 != 0)
        K = [0.01, 0.03]
        L = 255
        if len(K) == 2:
            if K[0] < 0 or K[1] < 0:
                return float('-inf'), float('-inf')
        else:
            return float('-inf'), float('-inf')

    elif nargin == 4:
        window = gkernel(1.5, sw)
        K = [0.01, 0.03]
        L = 255
        if len(K) == 2:
            if K[0] < 0 or K[1] < 0:
                return float('-inf'), float('-inf')
        else:
            return float('-inf'), float('-inf')

    C1 = (K[0] * L) ** 2
    C2 = (K[1] * L) ** 2
    window = window / np.sum(window)

    mu1 = convolve(img1, window, mode='same')
    mu2 = convolve(img2, window, mode='same')

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = convolve(img1 * img1, window, mode='same') - mu1_sq
    sigma2_sq = convolve(img2 * img2, window, mode='same') - mu2_sq
    sigma12 = convolve(img1 * img2, window, mode='same') - mu1_mu2

    if (C1 > 0) and (C2 > 0):
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    else:
        numerator1 = 2 * mu1_mu2 + C1
        numerator2 = 2 * sigma12 + C2
        denominator1 = mu1_sq + mu2_sq + C1
        denominator2 = sigma1_sq + sigma2_sq + C2
        
        ssim_map = np.ones(mu1.shape)
        index = (denominator1 * denominator2 > 0)
        ssim_map[index] = (numerator1[index] * numerator2[index]) / (denominator1[index] * denominator2[index])
        
        index = (denominator1 != 0) & (denominator2 == 0)
        ssim_map[index] = numerator1[index] / denominator1[index]

    temp = np.zeros(img1.shape)
    temp[ind] = 1
    iind = np.where(temp == 0)
    ssim_map[iind] = 1.0
    
    mssim = np.mean(ssim_map[ind])
    return mssim

# ==========================================
# PYTORCH SSIM FUNCTIONS (Optional/Alternative)
# ==========================================
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
def _ssim_3D(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv3d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv3d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv3d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim3D(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim_3D(img1, img2, window, window_size, channel, size_average)

def compute_ssim_numpy(img1_np, img2_np, window_size=11, size_average=True):
    img1_t = torch.from_numpy(img1_np).float()
    img2_t = torch.from_numpy(img2_np).float()

    if img1_t.ndimension() == 3:
        img1_t = img1_t.unsqueeze(0).unsqueeze(0)
        img2_t = img2_t.unsqueeze(0).unsqueeze(0)
    elif img1_t.ndimension() == 4:
        img1_t = img1_t.unsqueeze(1)
        img2_t = img2_t.unsqueeze(1)

    if torch.cuda.is_available():
        img1_t = img1_t.cuda()
        img2_t = img2_t.cuda()

    with torch.no_grad():
        return ssim3D(img1_t, img2_t, window_size=window_size, size_average=size_average)
