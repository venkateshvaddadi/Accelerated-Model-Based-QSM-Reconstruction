import torch
import torch.nn as nn
import torch.fft

# Global configuration variable from your setup
NO_CHANNELS = 32

class ConvReLUBlock(nn.Module):
    def __init__(self, channels=NO_CHANNELS):
        super().__init__()
        self.conv = nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.conv(x))

class BasicBlock(nn.Module):
    def __init__(self, inplanes=NO_CHANNELS, planes=NO_CHANNELS):
        super().__init__()
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
        return self.relu(out)

class WBasicBlock(nn.Module):
    def __init__(self, inplanes=NO_CHANNELS, planes=NO_CHANNELS, dropout_rate=0.5):
        super().__init__()
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
        return self.relu(out)

class WideResNet(nn.Module):
    def __init__(self, channels=NO_CHANNELS):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.gen = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            self._make_layer(WBasicBlock, 1),
            nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
        )
                
    def _make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x_input):
        return self.gen(x_input)

class ReducedRankExtrapolationQSM(nn.Module):
    """
    Reduced Rank Extrapolation (RRE) accelerated QSM reconstruction.
    Based on Awasthi et al., JBO 2018. Physics variant: D^T (D x - y).
    """
    def __init__(self, refiner_network=None, num_iters=12, k_order=3, rho=0.5):
        super().__init__()
        self.refiner = refiner_network
        self.num_iters = num_iters
        self.k_order = k_order
        self.rho = nn.Parameter(torch.tensor(rho, dtype=torch.float32))
        self.L = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, local_field, dipole_kernel):
        x_curr = torch.zeros_like(local_field)
        num_cycles = self.num_iters // self.k_order

        for _ in range(num_cycles):
            sequence = [x_curr]
            for _ in range(self.k_order):
                x_next = self.step_logic(sequence[-1], local_field, dipole_kernel)
                sequence.append(x_next)

            deltas = [(sequence[i + 1] - sequence[i]).reshape(-1) for i in range(self.k_order)]
            U = torch.stack(deltas, dim=1)

            try:
                Q, R = torch.linalg.qr(U, mode='reduced')
                ones = torch.ones(self.k_order, 1, device=U.device)
                y = torch.linalg.lstsq(R.T, ones).solution
                d = torch.linalg.lstsq(R, y).solution
            except RuntimeError:
                UTU = U.T @ U + 1e-6 * torch.eye(self.k_order, device=U.device)
                ones = torch.ones(self.k_order, 1, device=U.device)
                d = torch.linalg.solve(UTU, ones)

            gamma = d / torch.sum(d)
            x_extrapolated = torch.zeros_like(x_curr)

            for j in range(self.k_order + 1):
                weight = gamma[j] if j < self.k_order else (1 - torch.sum(gamma))
                x_extrapolated += weight * sequence[j]

            x_curr = x_extrapolated

        return x_curr

    def step_logic(self, x, y, dk):
        z = (1 - self.rho) * x + self.rho * self.refiner(x) if self.refiner is not None else x
        residual = self.dipole_conv(z, dk) - y
        grad = self.dipole_conv(residual, dk)
        return z - (1.0 / self.L) * grad

    def dipole_conv(self, x, kernel):
        return torch.real(torch.fft.ifftn(torch.fft.fftn(x) * kernel))

import torch
import torch.nn as nn
import torch.fft

class MomentumNetQSM(nn.Module):
    """
    Momentum-based Quantitative Susceptibility Mapping (QSM) Reconstruction Network.
    Implements a model-based iterative reconstruction unrolling with Nesterov-like acceleration.
    """
    def __init__(self, num_iters=10, rho=0.5):
        super().__init__()
        self.num_iters = num_iters
        self.rho = rho  # Relaxation parameter (reserved for future image refining networks)
        self.L = nn.Parameter(torch.tensor(1.0))  # Learnable majorizer step size

    def forward(self, local_field, dipole_kernel):
        """
        Args:
            local_field: Measured tissue phase field (y)
            dipole_kernel: 3D Dipole kernel tensor in the Fourier domain (D)
            
        Returns:
            x_curr: Reconstructed susceptibility distribution map (chi)
        """
        # x: Susceptibility map (chi), y: Local field
        x_curr = torch.zeros_like(local_field)
        x_prev = torch.zeros_like(local_field)
        
        # Initialization of momentum parameters
        theta_prev = 1.0
        
        for i in range(self.num_iters):  #
            # NOTE: Image Refining Module (e.g., WideResNet / U-Net Refiner) can be linked here
            # z = (1 - self.rho) * x_curr + self.rho * self.refiner(x_curr)
            
            # --- Extrapolation Module (Momentum computation) ---
            theta_curr = (1 + (1 + 4 * theta_prev**2)**0.5) / 2  #
            m = (theta_prev - 1) / theta_curr  # Calculate momentum coefficient
            
            # Compute extrapolated target coordinate
            x_hat = x_curr + m * (x_curr - x_prev)  #
            
            # --- MBIR Module (QSM Forward Physics Gradient Descent Step) ---
            # Forward physics matrix calculation: grad = D^T * (D * x_hat - local_field)
            residual = self.dipole_conv(x_hat, dipole_kernel) - local_field  #
            grad = self.dipole_conv(residual, dipole_kernel)  # D is a self-adjoint operator
            
            # Proximal gradient descent projection step
            x_next = x_hat - (1.0 / self.L) * grad  #
            
            # Cache iteration state updates
            x_prev = x_curr  #
            x_curr = x_next  #
            theta_prev = theta_curr  #
            
        return x_curr  #

    def dipole_conv(self, x, kernel):
        """Physics forward operations handled inside the Fourier Domain."""
        return torch.real(torch.fft.ifftn(torch.fft.fftn(x) * kernel))  #