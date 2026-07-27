"""EDM diffusion (ported from DiffGAD) for residual-augmented latent z."""

from __future__ import annotations

import math
from typing import Callable, Optional, Union

import torch
import torch.nn as nn

ModuleType = Union[str, Callable[..., nn.Module]]

SIGMA_MIN = 0.002
SIGMA_MAX = 80.0
RHO = 7.0
S_CHURN = 0.0  # disable stochastic churn for deterministic inference
S_MIN = 0.0
S_MAX = float("inf")
S_NOISE = 1.0


class EDMLoss:
    def __init__(
        self,
        P_mean: float = -1.2,
        P_std: float = 1.2,
        sigma_data: float = 0.5,
        hid_dim: int = 100,
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.hid_dim = hid_dim

    def __call__(self, denoise_fn, data, proto=None, proto_alpha=None):
        rnd_normal = torch.randn(data.shape[0], device=data.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        y = data
        n = torch.randn_like(y) * sigma.unsqueeze(1)
        D_yn = denoise_fn(y + n, sigma, proto, proto_alpha)

        loss = weight.unsqueeze(1) * ((D_yn - y) ** 2)
        reconstruction_errors = (D_yn - y) ** 2
        score = torch.sqrt(torch.sum(reconstruction_errors, 1))
        return loss, score, D_yn


class PositionalEmbedding(nn.Module):
    def __init__(self, num_channels: int, max_positions: int = 10000, endpoint: bool = False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freqs = torch.arange(
            start=0,
            end=self.num_channels // 2,
            dtype=torch.float32,
            device=x.device,
        )
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        return torch.cat([x.cos(), x.sin()], dim=1)


class MLPDiffusion(nn.Module):
    """Free-only MLP denoiser (proto path kept for DiffGAD API compatibility)."""

    def __init__(self, d_in: int, dim_t: int = 512):
        super().__init__()
        self.dim_t = dim_t
        self.proj = nn.Linear(d_in, dim_t)
        self.mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, d_in),
        )
        self.map_noise = PositionalEmbedding(num_channels=dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t),
        )
        self.proto_proj = nn.Linear(d_in, dim_t)

    def forward(self, x, noise_labels, proto=None, proto_alpha=None):
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
        emb = self.time_embed(emb)
        if proto is None:
            x = self.proj(x) + emb
        else:
            alpha = 0.0 if proto_alpha is None else float(proto_alpha)
            x = self.proj(x) + emb + alpha * self.proto_proj(proto)
        return self.mlp(x)


class Precond(nn.Module):
    def __init__(
        self,
        denoise_fn,
        hid_dim: int,
        sigma_min: float = 0.0,
        sigma_max: float = float("inf"),
        sigma_data: float = 0.5,
    ):
        super().__init__()
        self.hid_dim = hid_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.denoise_fn_F = denoise_fn

    def forward(self, x, sigma, proto=None, proto_alpha=None):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1)

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.denoise_fn_F((c_in * x).to(torch.float32), c_noise.flatten(), proto, proto_alpha)
        return c_skip * x + c_out * F_x.to(torch.float32)

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


class DiffusionModel(nn.Module):
    def __init__(
        self,
        denoise_fn,
        hid_dim: int,
        P_mean: float = -1.2,
        P_std: float = 1.2,
        sigma_data: float = 0.5,
    ):
        super().__init__()
        self.denoise_fn_D = Precond(denoise_fn, hid_dim, sigma_data=sigma_data)
        self.loss_fn = EDMLoss(P_mean, P_std, sigma_data, hid_dim=hid_dim)

    def forward(self, x, proto=None, proto_alpha=None):
        loss, score, reconstructed = self.loss_fn(self.denoise_fn_D, x, proto, proto_alpha)
        return loss.mean(-1).mean(), score, reconstructed


def _build_t_steps(net: Precond, num_steps: int, device: torch.device) -> torch.Tensor:
    sigma_min = max(SIGMA_MIN, float(net.sigma_min))
    sigma_max = min(SIGMA_MAX, float(net.sigma_max) if math.isfinite(float(net.sigma_max)) else SIGMA_MAX)
    if num_steps <= 1:
        t_steps = torch.tensor([sigma_max], dtype=torch.float32, device=device)
    else:
        step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)
        t_steps = (
            sigma_max ** (1 / RHO)
            + step_indices / (num_steps - 1) * (sigma_min ** (1 / RHO) - sigma_max ** (1 / RHO))
        ) ** RHO
    return torch.cat([net.round_sigma(t_steps).to(device=device, dtype=torch.float32), torch.zeros(1, device=device)])


def sample_step(net, num_steps, i, t_cur, t_next, x_next, proto=None, proto_alpha=None):
    x_cur = x_next
    gamma = min(S_CHURN / max(num_steps, 1), math.sqrt(2) - 1) if S_MIN <= t_cur <= S_MAX else 0.0
    t_hat = net.round_sigma(t_cur + gamma * t_cur)
    x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_NOISE * torch.randn_like(x_cur)

    denoised = net(x_hat, t_hat, proto, proto_alpha).to(torch.float32)
    d_cur = (x_hat - denoised) / t_hat
    x_next = x_hat + (t_next - t_hat) * d_cur
    if i < num_steps - 1:
        denoised = net(x_next, t_next, proto, proto_alpha).to(torch.float32)
        d_prime = (x_next - denoised) / t_next
        x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next


def sample_dm(net, noise, num_steps: int, proto=None, proto_alpha=None):
    """EDM sampler from pure noise (free-only; no prototype guidance)."""
    num_steps = max(int(num_steps), 1)
    t_steps = _build_t_steps(net, num_steps, noise.device)
    z = noise.to(torch.float32) * t_steps[0]
    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            z = sample_step(net, num_steps, i, t_cur, t_next, z, proto, proto_alpha)
    return z
