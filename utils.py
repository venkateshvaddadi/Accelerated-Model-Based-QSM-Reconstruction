import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time

# --- MATLAB Timing Emulation ---
START_TIME = 0

def tic():
    global START_TIME
    START_TIME = time.time()

def toc():
    if START_TIME == 0:
        print("Toc: start time not set")
    else:
        print(f"{time.time() - START_TIME:.4f} seconds")

# --- QSM Physics Utilities ---
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
    
    return torch.tensor(np.float32(D)).unsqueeze(dim=0)

def sobel_kernel():
    s = torch.FloatTensor([
        [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]
    ])
    sx = s
    sy = s.permute(1, 2, 0)
    sz = s.permute(2, 0, 1)
    return torch.stack([sx, sy, sz]).unsqueeze(1)

# --- Loss Functions ---
def gradient_loss(gen_chi):
    dx = gen_chi[:, :, 1:, :, :] - gen_chi[:, :, :-1, :, :]
    dy = gen_chi[:, :, :, 1:, :] - gen_chi[:, :, :, :-1, :]
    dz = gen_chi[:, :, :, :, 1:] - gen_chi[:, :, :, :, :-1]
    return torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy)) + torch.mean(torch.abs(dz))

def total_loss_l1(chi, y, b, d, m, sobel):    
    def _l1error(x1, x2):
        return torch.mean(torch.abs(x1 - x2))

    chi_fourier = torch.fft.fftn(chi, dim=[2, 3, 4])
    b_hat = torch.real(torch.fft.ifftn(chi_fourier * d, dim=[2, 3, 4]))
    
    b = b * m
    b_hat = b_hat * m    
    
    loss_model = _l1error(b, b_hat)
    difference = F.conv3d((y - chi).float(), sobel.float(), padding=1)
    loss_grad = torch.mean(torch.abs(difference))
    
    return 0.9 * loss_model + 0.1 * loss_grad

# --- Visualization ---
def plot_qsm_comparison(reconstruction, ground_truth, mask, title="QSM Reconstruction"):
    recon = np.squeeze(reconstruction) * np.squeeze(mask)
    gt = np.squeeze(ground_truth) * np.squeeze(mask)
    
    mid_slice = recon.shape[2] // 2
    recon_slice = recon[:, :, mid_slice]
    gt_slice = gt[:, :, mid_slice]
    error_slice = np.abs(gt_slice - recon_slice)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    vmin, vmax = -0.1, 0.1 

    im1 = axes[0].imshow(gt_slice, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0].set_title("Ground Truth (COSMOS)")
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(recon_slice, cmap='gray', vmin=vmin, vmax=vmax)
    axes[1].set_title("Reconstruction")
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    im3 = axes[2].imshow(error_slice, cmap='hot', vmin=0, vmax=0.05)
    axes[2].set_title("Absolute Error Map")
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


import torch
import numpy as np
import matplotlib.pyplot as plt
import time

# --- MATLAB-style Context Timers ---
START_TIME = 0

def tic():
    """Initializes context tracking epoch timer stamps[cite: 115]."""
    global START_TIME
    START_TIME = time.time()  # [cite: 115]

def toc():
    """Outputs time calculation deltas[cite: 115]."""
    if START_TIME == 0:
        print("Toc: start time not set")  # [cite: 115]
    else:
        print(f"{time.time() - START_TIME:.4f} seconds")  # [cite: 115]

# --- Analytical QSM Physics Utilities ---
def dipole_kernel(matrix_size, voxel_size, B0_dir=[0, 0, 1]):
    """
    Generates a 3D Dipole Kernel field distribution in the Fourier Domain[cite: 104].
    Compatible with NumPy 2.0+[cite: 104].
    """
    [Y, X, Z] = np.meshgrid(
        np.linspace(-int(matrix_size[1]/2), int(matrix_size[1]/2)-1, matrix_size[1]),  # [cite: 104]
        np.linspace(-int(matrix_size[0]/2), int(matrix_size[0]/2)-1, matrix_size[0]),  # [cite: 104]
        np.linspace(-int(matrix_size[2]/2), int(matrix_size[2]/2)-1, matrix_size[2])   # [cite: 104]
    )
    
    X = X / (matrix_size[0]) * voxel_size[0]  # [cite: 104]
    Y = Y / (matrix_size[1]) * voxel_size[1]  # [cite: 104]
    Z = Z / (matrix_size[2]) * voxel_size[2]  # [cite: 104]
    
    # Calculate Dipole Kernel formulation: D = 1/3 - (kz^2 / k^2) [cite: 105]
    numerator = np.square(X * B0_dir[0] + Y * B0_dir[1] + Z * B0_dir[2])  # [cite: 105]
    denominator = np.square(X) + np.square(Y) + np.square(Z) + np.finfo(float).eps  # [cite: 105]
    D = 1/3 - np.divide(numerator, denominator)  # [cite: 105]
    D = np.where(np.isnan(D), 0, D)  # [cite: 105]

    # Center shifting operations [cite: 106]
    D = np.roll(D, int(np.floor(matrix_size[0]/2)), axis=0)  # [cite: 106]
    D = np.roll(D, int(np.floor(matrix_size[1]/2)), axis=1)  # [cite: 106]
    D = np.roll(D, int(np.floor(matrix_size[2]/2)), axis=2)  # [cite: 106]
    
    return torch.tensor(np.float32(D)).unsqueeze(dim=0)  # Return batch-ready tensor [cite: 106]

# --- Visualization Module ---
def plot_qsm_comparison(reconstruction, ground_truth, mask, title="QSM Reconstruction"):
    """
    Slices and evaluates center matrices comparing estimations against target COSMOS fields[cite: 126, 128].
    """
    recon = np.squeeze(reconstruction) * np.squeeze(mask)  # [cite: 126]
    gt = np.squeeze(ground_truth) * np.squeeze(mask)  # [cite: 126]
    
    mid_slice = recon.shape[2] // 2  # [cite: 127]
    recon_slice = recon[:, :, mid_slice]  # [cite: 127]
    gt_slice = gt[:, :, mid_slice]  # [cite: 127]
    error_slice = np.abs(gt_slice - recon_slice)  # [cite: 127]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # [cite: 128]
    vmin, vmax = -0.1, 0.1   # Clipping threshold filters [cite: 128]

    im1 = axes[0].imshow(gt_slice, cmap='gray', vmin=vmin, vmax=vmax)  # [cite: 128]
    axes[0].set_title("Ground Truth (COSMOS)")  # [cite: 128]
    axes[0].axis('off')  # [cite: 128]
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)  # [cite: 128]

    im2 = axes[1].imshow(recon_slice, cmap='gray', vmin=vmin, vmax=vmax)  # [cite: 128]
    axes[1].set_title("Reconstruction")  # [cite: 128]
    axes[1].axis('off')  # [cite: 128]
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)  # [cite: 128]

    im3 = axes[2].imshow(error_slice, cmap='hot', vmin=0, vmax=0.05)  # [cite: 129]
    axes[2].set_title("Absolute Error Map")  # [cite: 129]
    axes[2].axis('off')  # [cite: 129]
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)  # [cite: 129]

    plt.suptitle(title)  # [cite: 129]
    plt.tight_layout()  # [cite: 129]
    plt.show()  # [cite: 129]