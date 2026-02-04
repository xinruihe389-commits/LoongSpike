import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, List, Tuple

from src.models.nn import DropoutNd
from src.models.spike.neuron import registry
from src.models.spike.surrogate import piecewise_quadratic_surrogate
from src.models.sequence.kernels.ssm import SSMKernelDiag, SSMKernelFrac


class LoongSpikeKernel(nn.Module):
    """Fractional SSM kernel with manually specified omega_i and eta_i for LoongSpike model."""
    
    def __init__(
        self,
        d_model,
        N: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        channels: int = 1,
        lr: Optional[float] = None,
        # Fractional SSM parameters
        frac_M: int = 2,
        manual_omega: Optional[torch.Tensor] = None,
        manual_eta: Optional[torch.Tensor] = None,
        # Whether to make omega_i and eta_i learnable
        learnable_omega_eta: bool = True,
        # Use loop-based implementation (memory efficient)
        use_loop: bool = False,
        # Constraints for learnable parameters (for numerical stability)
        omega_min: float = 1e-6,
        omega_max: float = 100.0,
        eta_min: float = 1e-6,
        eta_max: float = 10.0,
    ):
        super().__init__()
        
        # Generate dt
        if lr is None:
            lr = 0.001
        else:
            lr = min(lr, 0.001)
        
        H = d_model
        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        
        self.N = N
        self.frac_M = frac_M
        self.use_loop = use_loop
        self.learnable_omega_eta = learnable_omega_eta
        # Store constraints for learnable parameters
        self.omega_min = omega_min
        self.omega_max = omega_max
        self.eta_min = eta_min
        self.eta_max = eta_max
        
        # Initialize C, A parameters (same as standard S4D)
        C = torch.randn(channels, H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        self.register("log_dt", log_dt, lr)
        
        log_A_real = torch.log(0.5 * torch.ones(H, N // 2))
        A_imag = math.pi * repeat(torch.arange(N // 2), "n -> h n", h=H)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)
        
        # Initialize omega_i and eta_i
        if manual_omega is not None and manual_eta is not None:
            # Use manually specified values
            # Convert to tensor if needed and ensure shape is (M,)
            if not isinstance(manual_omega, torch.Tensor):
                manual_omega = torch.tensor(manual_omega, dtype=torch.float32)
            if not isinstance(manual_eta, torch.Tensor):
                manual_eta = torch.tensor(manual_eta, dtype=torch.float32)
            
            # Ensure shape is (M,)
            if manual_omega.dim() > 1:
                manual_omega = manual_omega.flatten()
            if manual_eta.dim() > 1:
                manual_eta = manual_eta.flatten()
            
            assert manual_omega.shape == (frac_M,), f"manual_omega shape mismatch: {manual_omega.shape} vs expected ({frac_M},)"
            assert manual_eta.shape == (frac_M,), f"manual_eta shape mismatch: {manual_eta.shape} vs expected ({frac_M},)"
            
            if learnable_omega_eta:
                # For learnable parameters, use unconstrained parameters and apply constraints in forward
                omega_clamped = manual_omega.clamp(omega_min, omega_max)
                if omega_max > omega_min:
                    omega_normalized = (omega_clamped - omega_min) / (omega_max - omega_min)
                    omega_normalized = omega_normalized.clamp(1e-6, 1 - 1e-6)
                    self.omega_logit = nn.Parameter(torch.logit(omega_normalized))
                else:
                    # If omega_min == omega_max, use a fixed value
                    self.omega_logit = nn.Parameter(torch.zeros_like(omega_clamped))
                # eta: use sigmoid to map to [eta_min, eta_max]
                eta_clamped = manual_eta.clamp(eta_min, eta_max)
                if eta_max > eta_min:
                    eta_normalized = (eta_clamped - eta_min) / (eta_max - eta_min)
                    eta_normalized = eta_normalized.clamp(1e-6, 1 - 1e-6)
                    self.eta_logit = nn.Parameter(torch.logit(eta_normalized))
                else:
                    self.eta_logit = nn.Parameter(torch.zeros_like(eta_clamped))
            else:
                self.register_buffer("omega", manual_omega)
                self.register_buffer("eta", manual_eta)
        else:
            # Use default values based on frac_M (shape: (M,))
            if frac_M == 1:
                # M=1: baseline case
                default_omega = torch.zeros(1)
                default_eta = torch.ones(1)
            elif frac_M == 2:
                default_omega = torch.tensor([0.0, 0.1])  # Smaller omega_2 for slower decay
                default_eta = torch.tensor([1.0, 0.1])   # First term dominates
            else:
                # M>=3: extend with more terms
                # Default: evenly spaced omega from 0 to 0.1, uniform eta
                omega_values = torch.linspace(0.0, 0.1, frac_M)
                default_omega = omega_values
                default_eta = torch.ones(frac_M) / frac_M
            
            if learnable_omega_eta:
                # For learnable parameters, use unconstrained parameters and apply constraints in forward
                # omega: use sigmoid to map to [omega_min, omega_max]
                omega_clamped = default_omega.clamp(omega_min, omega_max)
                if omega_max > omega_min:
                    omega_normalized = (omega_clamped - omega_min) / (omega_max - omega_min)
                    omega_normalized = omega_normalized.clamp(1e-6, 1 - 1e-6)
                    self.omega_logit = nn.Parameter(torch.logit(omega_normalized))
                else:
                    # If omega_min == omega_max, use a fixed value
                    self.omega_logit = nn.Parameter(torch.zeros_like(omega_clamped))
                # eta: use sigmoid to map to [eta_min, eta_max]
                eta_clamped = default_eta.clamp(eta_min, eta_max)
                if eta_max > eta_min:
                    eta_normalized = (eta_clamped - eta_min) / (eta_max - eta_min)
                    eta_normalized = eta_normalized.clamp(1e-6, 1 - 1e-6)
                    self.eta_logit = nn.Parameter(torch.logit(eta_normalized))
                else:
                    self.eta_logit = nn.Parameter(torch.zeros_like(eta_clamped))
            else:
                self.register_buffer("omega", default_omega)
                self.register_buffer("eta", default_eta)
    
    def forward(self, L: int):
        """Compute fractional SSM convolution kernel."""
        # Materialize original diagonal SSM parameters
        dt = torch.exp(self.log_dt)  # (H)
        C = torch.view_as_complex(self.C)  # (C, H, N_state)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H, N_state)
        
        device = A.device
        base_dtype = A.real.dtype
        H, N_state = A.shape
        C_channels = C.shape[0]
        M = self.frac_M
        
        # Get omega and eta (apply constraints if learnable)
        if self.learnable_omega_eta:
            # omega: apply sigmoid constraint to [omega_min, omega_max]
            omega_normalized = torch.sigmoid(self.omega_logit)  # (M,) in [0, 1]
            if self.omega_max > self.omega_min:
                omega = self.omega_min + omega_normalized * (self.omega_max - self.omega_min)  # (M,)
            else:
                omega = torch.full_like(omega_normalized, self.omega_min)  # (M,)
            # eta: apply sigmoid constraint to [eta_min, eta_max]
            eta_normalized = torch.sigmoid(self.eta_logit)  # (M,) in [0, 1]
            if self.eta_max > self.eta_min:
                eta = self.eta_min + eta_normalized * (self.eta_max - self.eta_min)  # (M,)
            else:
                eta = torch.full_like(eta_normalized, self.eta_min)  # (M,)
        else:
            omega = self.omega.to(device).to(base_dtype)  # (M,)
            eta = self.eta.to(device).to(base_dtype)  # (M,)
        
        # Normalize eta to ensure sum(eta_i) = 1
        #eta = eta / (eta.sum() + 1e-8)  # (M,)
        
        if self.use_loop:
            K_total = torch.zeros(C_channels, H, L, dtype=base_dtype, device=device)  # (C, H, L)
            
            for i in range(M):
                # Get omega_i and eta_i for the i-th term (scalars, will broadcast)
                omega_i = omega[i]  # scalar
                eta_i = eta[i]  # scalar
                
                # Compute A_frac_i = -omega_i + eta_i * A
                # omega_i and eta_i are scalars, PyTorch will broadcast to (H, N_state)
                A_frac_i = -omega_i + eta_i * A  # (H, N_state)
                
                # Compute C_frac_i = eta_i * C
                C_frac_i = eta_i * C  # (C, H, N_state)
                
                # Discretize
                dtA_i = A_frac_i * dt.unsqueeze(-1)  # (H, N_state)
                exp_dtA_i = torch.exp(dtA_i)  # (H, N_state)
                
                # Handle A_frac_i ≈ 0 case
                small_mask_i = torch.abs(A_frac_i) < 1e-6
                C_disc_i = C_frac_i * (exp_dtA_i - 1.0) / (A_frac_i + 1e-8)  # (C, H, N_state)
                if small_mask_i.any():
                    dt_expanded_i = dt.unsqueeze(-1).unsqueeze(0).expand(C_channels, H, N_state)
                    C_disc_small_i = C_frac_i * dt_expanded_i
                    C_disc_i = torch.where(small_mask_i.unsqueeze(0), C_disc_small_i, C_disc_i)
                
                # Compute convolution kernel contribution for the i-th term
                K_i = dtA_i.unsqueeze(-1) * torch.arange(L, device=device, dtype=base_dtype)  # (H, N_state, L)
                exp_K_i = torch.exp(K_i)  # (H, N_state, L)
                
                # Accumulate contribution
                K_i_contribution = 2 * torch.einsum("chn, hnl -> chl", C_disc_i, exp_K_i).real  # (C, H, L)
                K_total = K_total + K_i_contribution
                
                # Release memory (important for memory efficiency)
                del K_i, exp_K_i, C_disc_i, dtA_i, A_frac_i, C_frac_i, exp_dtA_i
            
            K = K_total
        else:
            # Vectorized implementation: omega and eta are (M,), need to broadcast
            A_exp = A.unsqueeze(1).expand(H, M, N_state)  # (H, M, N_state)
            omega_exp = omega.unsqueeze(0).unsqueeze(-1)  # (1, M, 1) -> broadcasts to (H, M, 1)
            eta_exp = eta.unsqueeze(0).unsqueeze(-1)  # (1, M, 1) -> broadcasts to (H, M, 1)
            
            # Compute A_frac = -omega + eta * A
            A_frac = -omega_exp + eta_exp * A_exp  # (H, M, N_state)
            
            # Compute C_frac = eta * C
            C_exp = C.unsqueeze(2).expand(C_channels, H, M, N_state)  # (C, H, M, N_state)
            C_frac = eta_exp.unsqueeze(0) * C_exp  # (1, H, M, 1) * (C, H, M, N_state) -> (C, H, M, N_state)
            
            # Reshape
            A_frac = A_frac.reshape(H, M * N_state)  # (H, M*N_state)
            C_frac = C_frac.reshape(C_channels, H, M * N_state)  # (C, H, M*N_state)
            
            # Compute convolution kernel
            dtA = A_frac * dt.unsqueeze(-1)  # (H, M*N_state)
            exp_dtA = torch.exp(dtA)  # (H, M*N_state)
            
            # Handle A_frac ≈ 0 case
            small_mask = torch.abs(A_frac) < 1e-6
            C_disc = C_frac * (exp_dtA.unsqueeze(0) - 1.0) / (A_frac.unsqueeze(0) + 1e-8)  # (C, H, M*N_state)
            if small_mask.any():
                dt_expanded = dt.unsqueeze(-1).unsqueeze(0).expand(C_channels, H, A_frac.shape[-1])
                C_disc_small = C_frac * dt_expanded
                C_disc = torch.where(small_mask.unsqueeze(0), C_disc_small, C_disc)
            
            K_all = dtA.unsqueeze(-1) * torch.arange(L, device=device, dtype=base_dtype)  # (H, M*N_state, L)
            exp_K_all = torch.exp(K_all)  # (H, M*N_state, L)
            K = 2 * torch.einsum("chn, hnl -> chl", C_disc, exp_K_all).real  # (C, H, L)
            
            del exp_K_all, C_disc, dtA, A_frac, C_frac, exp_dtA
        
        return K, None
    
    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""
        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))
            optim = {"weight_decay": 0.0}
            if lr is not None:
                optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class LoongSpike(nn.Module):
    """LoongSpike layer: Spiking neural network with fractional SSM kernel."""
   
    def __init__(
        self,
        d_model,
        neuron="sdn",
        learnable_vth=True,
        shared_vth=False,
        d_state=64,
        dropout=0.0,
        transposed=True,
        bidirectional=False,
        channels=1,
        trainable_B=False,
        **kernel_args,
    ):
        super().__init__()
        
        self.h = d_model
        self.n = d_state
        self.transposed = transposed
        self.learnable_vth = learnable_vth
        self.bidirectional = bidirectional
        self.D = nn.Parameter(torch.randn(channels, self.h))
        
        if learnable_vth:
            if shared_vth:
                self.ln_vth = nn.Parameter(torch.zeros(1))
            else:
                self.ln_vth = nn.Parameter(torch.zeros(d_model, 1))
        
        if bidirectional:
            channels *= 2
        
        # SSM Kernel
        if trainable_B:
            self.kernel = SSMKernelFrac(d_model=self.h, d_state=self.n, channels=channels, init="diag-lin", **kernel_args)
        else:
            self.kernel = LoongSpikeKernel(
                self.h,
                N=self.n,
                channels=channels,
                **kernel_args,
            )
        
        self.neuron = registry[neuron](piecewise_quadratic_surrogate())
        
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()
        
        # position-wise output transform to mix features
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2 * self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )
    
    def forward(self, u, **kwargs):
        """Input and output shape (B, H, L)"""
        if not self.transposed:
            u = u.transpose(-1, -2)
        L = u.size(-1)
        
        # Compute SSM Kernel
        k, _ = self.kernel(L=L)  # (C, H, L)
        if self.bidirectional:
            k0, k1 = rearrange(k, "(s c) h l -> s c h l", s=2)
            k = F.pad(k0, (0, L)) + F.pad(k1.flip(-1), (L, 0))
        
        # Convolution
        k_f = torch.fft.rfft(k, n=2 * L)  # (C, H, L)
        u_f = torch.fft.rfft(u, n=2 * L)  # (B, H, L)
        
        y = torch.einsum("bhl,chl->bchl", u_f, k_f)
        del u_f, k_f
        
        y = torch.fft.irfft(y, n=2 * L)[..., :L]  # (B, C, H, L)
        
        # Compute D term in state space equation - essentially a skip connection
        y = y + torch.einsum("bhl,ch->bchl", u, self.D)
        
        y = rearrange(y, "b c h l -> b (c h) l")
        
        if self.learnable_vth:
            y = y / torch.exp(self.ln_vth)
        
        y = self.dropout(self.neuron(y))
        y = self.output_linear(y)
        if not self.transposed:
            y = y.transpose(-1, -2)
        return (
            y,
            None,
        )


class LoongSpikingSSM(nn.Module):
    """
    LoongSpike Spiking SSM: Multi-layer spiking neural network with fractional SSM layers.
    """
    
    def __init__(
        self,
        d_model=256,
        n_layers=4,
        dropout=0.2,
        prenorm=False,
        norm="layer",
        layer=None,  # layer config
        **kwargs,
    ):
        super().__init__()
        
        self.prenorm = prenorm
        self.norm = norm
        
        # for dataset adaptability
        self.d_model = self.d_output = d_model
        
        # Stack LoongSpike layers as residual blocks
        self.loongspike_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.loongspike_layers.append(LoongSpike(d_model, dropout=dropout, transposed=True, **layer))
            if norm == "batch":
                self.norms.append(nn.BatchNorm1d(d_model))
            elif norm == "layer":
                self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(DropoutNd(dropout))
    
    def forward(self, x, **kwargs):
        """
        Input x is shape (B, L, d_input)
        """
        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.loongspike_layers, self.norms, self.dropouts):
            z = x
            # Prenorm
            if self.prenorm:
                z = norm(z) if self.norm == "batch" else norm(z.transpose(-1, -2)).transpose(-1, -2)
            
            # Apply LoongSpike block
            z, _ = layer(z)
            
            # Dropout on the output of the LoongSpike block
            z = dropout(z)
            
            # Residual connection
            x = z + x
            
            if not self.prenorm:
                # Postnorm
                z = norm(z) if self.norm == "batch" else norm(z.transpose(-1, -2)).transpose(-1, -2)
        
        x = x.transpose(-1, -2)
        
        return x, None

