import os
import scipy.io
import numpy as np
import pandas as pd
from datetime import datetime
import torch

# Internal Imports
from models import MomentumNetQSM
from utils import dipole_kernel, tic, toc, plot_qsm_comparison
from modules.metrics import compute_psnr, compute_rmse, compute_hfen, compute_ssim_numpy

# --- Runtime Configuration & Device ---
device_id = 0
device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
base_dir = "savedModels/MOMENTUM_QSM_EXPERIMENTS/"
model_name = "MomentumQSM"

batch_size = 1
num_iters = 9
rho = 0.5
lr = 1e-4

# Build Experiment Dir
timestamp = datetime.now().strftime("%b_%d_%H_%M_%S")
exp_name = f"{model_name}_K_{num_iters}"
exp_dir = os.path.join(base_dir, exp_name)
os.makedirs(exp_dir, exist_ok=True)

print("\n" + "="*50)
print(f"🚀 STARTING EXPERIMENT: {model_name}")
print("="*50)
print(f"📂 Save Directory:  {exp_dir}")
print(f"💻 Device:          {device}")
print("="*50 + "\n")

# --- Initialize Model & Optimization Modules ---
model = MomentumNetQSM(num_iters=num_iters, rho=rho).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# --- Patient Evaluation Loop ---
results = []
patients_list = list(range(1, 13))
orientations = list(range(1, 6))
raw_data_path = '../../QSM_data/data_for_experiments/given_data/raw_data_names_modified/'

for i in patients_list:
    for j in orientations:
        # Data Paths & Loading
        p_path = f"{raw_data_path}/patient_{i}"
        phs_raw = scipy.io.loadmat(f"{p_path}/phs{j}.mat")['phs']
        msk_raw = scipy.io.loadmat(f"{p_path}/msk{j}.mat")['msk']
        sus_raw = scipy.io.loadmat(f"{p_path}/cos{j}.mat")['cos']

        # Send Phase and Mask arrays to Engine
        phs = torch.tensor(phs_raw).float().unsqueeze(0).unsqueeze(0).to(device)
        msk = torch.tensor(msk_raw).float().unsqueeze(0).unsqueeze(0).to(device)

        # Dynamic Dipole Generation (Based on input shapes)
        temp_shape = phs.shape
        matrix_size = [temp_shape[2], temp_shape[3], temp_shape[4]]
        voxel_size = [1, 1, 1]
        dk = dipole_kernel(matrix_size, voxel_size, B0_dir=[0, 0, 1]).to(device)
        dk_rep = dk.repeat(batch_size, 1, 1, 1, 1)

        # Train Step Optimization
        tic()
        chi_pred = model(phs, dk_rep)
        toc()

        phi_pred = torch.fft.ifftn(dk_rep * torch.fft.fftn(chi_pred, dim=[2, 3, 4]), dim=[2, 3, 4])
        loss = torch.mean((phi_pred.real - phs) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        x_k_cpu = (chi_pred.real.detach().cpu().numpy()) * (msk.detach().cpu().numpy())

        # Metric Evaluations
        psnr_val = compute_psnr(x_k_cpu, sus_raw)
        rmse_val = compute_rmse(x_k_cpu, sus_raw)
        hfen_val = compute_hfen(x_k_cpu, sus_raw)

        sus_expanded = np.expand_dims(np.expand_dims(sus_raw, axis=0), axis=0)
        ssim_val = compute_ssim_numpy(x_k_cpu, sus_expanded)

        print({
            'patient': i, 'orientation': j,
            'SSIM': f'{ssim_val:.4f}', 'Loss': f'{loss.item():.2e}',
            'PSNR': f'{psnr_val:.2f}', 'RMSE': f'{rmse_val:.4f}', 'HFEN': f'{hfen_val:.4f}'
        })

        results.append({
            'patient': i, 'orientation': j, 'loss': loss.item(),
            'ssim': ssim_val.item(), 'psnr': psnr_val, 'rmse': rmse_val, 'hfen': hfen_val
        })

        # Save Predictions
        scipy.io.savemat(os.path.join(exp_dir, f"patient_{i}_orientation_{j}.mat"), {"chi_recon": x_k_cpu})

# --- Save Dataframes and Print Summary Logs ---
df = pd.DataFrame(results)
save_path = os.path.join(exp_dir, "momentum_qsm_results.csv")
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
plot_qsm_comparison(x_k_cpu, sus_raw, msk_raw, title=f"Patient {i} Orientation {j}")