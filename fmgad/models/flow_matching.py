import torch
import torch.nn as nn

from fmgad.losses import conditional_flow_matching_loss, flow_matching_loss


class FlowMatchingLoss:
    """Flow Matching Loss - learns velocity field instead of denoising"""
    def __init__(self, sigma_min=0.01, sigma_max=1.0, hid_dim=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.hid_dim = hid_dim
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def __call__(self, velocity_fn, data, proto=None, proto_alpha=None):
        """
        Flow Matching loss: learn velocity field v_t(x_t) that transports
        from noise distribution to data distribution

        Path: x_t = (1-t) * x_0 + t * x_1
        where x_0 = data, x_1 = noise
        True velocity: v_t = x_1 - x_0 = noise - data
        """
        batch_size = data.shape[0]
        device = data.device

        # Sample time uniformly in [0, 1]
        t = torch.rand(batch_size, device=device)

        # Sample noise
        noise = torch.randn_like(data)

        # Linear interpolation path: x_t = (1-t) * data + t * noise
        t_expanded = t.unsqueeze(1)  # [B, 1]
        x_t = (1 - t_expanded) * data + t_expanded * noise

        # True velocity: v_t = noise - data (constant along path)
        v_true = noise - data

        # Predicted velocity
        v_pred = velocity_fn(x_t, t, proto, proto_alpha)

        # L2 loss
        loss = torch.mean((v_pred - v_true) ** 2)

        # Reconstruction error for scoring (similar to diffusion)
        reconstruction_errors = (v_pred - v_true) ** 2
        score = torch.sqrt(torch.sum(reconstruction_errors, 1))

        # Reconstructed data point (for prototype computation)
        # Use the predicted velocity to estimate reconstruction
        # At t=0, we can reconstruct as: x_0 ≈ x_t - t * v_pred
        # But for scoring, we use the error directly
        reconstructed = data - v_pred  # Approximate reconstruction

        return loss, score, reconstructed


class MLPFlowMatching(nn.Module):
    """MLP velocity field for flow matching (v1-style improvements)."""
    def __init__(self, d_in, dim_t=512, cond_dim=None):
        super().__init__()
        self.dim_t = dim_t
        self.cond_dim = cond_dim

        # Time embedding (sinusoidal map)
        self.map_time = PositionalEmbedding(num_channels=dim_t)

        # MLP on time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )

        # Optional prototype / conditioning path
        if cond_dim is not None:
            self.map_proto = nn.Linear(cond_dim, dim_t, bias=False)
            self.proto_proj = nn.Linear(cond_dim, dim_t)

        # Project state x to embedding width
        self.proj_x = nn.Linear(d_in, dim_t)

        # Main MLP head: predict velocity vector
        self.mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, d_in),  # output dim matches input dim
        )

    def forward(self, x, t, context=None, proto_alpha=None):
        """
        Predict velocity field v_t(x_t).
        Args:
            x: current state [B, d]
            t: time [B] or scalar
            context: prototype context (optional) [B, cond_dim] or [1, cond_dim]
            proto_alpha: prototype weight (optional)
        Returns:
            velocity: predicted velocity [B, d]
        """
        # Normalize t to shape [B]
        if isinstance(t, (int, float)):
            t = torch.tensor([t], device=x.device, dtype=x.dtype).expand(x.shape[0])
        elif t.dim() == 0:
            t = t.unsqueeze(0).expand(x.shape[0])
        elif t.dim() > 1:
            t = t.squeeze(-1) if t.shape[-1] == 1 else t.flatten()

        # Clamp t to [0, 1]
        t = t.clamp(0.0, 1.0)

        # Time embedding (scaled time for richer positional signal)
        t_emb = self.map_time(t * 1000.0)
        t_emb = t_emb.reshape(t_emb.shape[0], 2, -1).flip(1).reshape(*t_emb.shape)
        t_emb = self.time_embed(t_emb)

        # Project x
        x_proj = self.proj_x(x)

        # Fuse optional prototype context
        if context is not None and self.cond_dim is not None:
            if proto_alpha is None:
                proto_alpha = 1.0
            proto_emb = self.proto_proj(context)
            # x + time + scaled prototype
            h = x_proj + t_emb + proto_alpha * proto_emb
        else:
            # Unconditional: x + time only
            h = x_proj + t_emb

        v_t = self.mlp(h)
        return v_t


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels // 2,
                             dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class FlowMatchingModel(nn.Module):
    """Flow-matching training wrapper."""
    def __init__(self, velocity_fn, hid_dim, sigma_min=0.01, sigma_max=1.0):
        super().__init__()
        self.hid_dim = hid_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.velocity_fn = velocity_fn

    def forward(self, x_1, proto=None, proto_alpha=None):
        """
        Forward: flow-matching loss plus a reconstruction-based anomaly score.

        Args:
            x_1: real node features [batch, hid_dim]
            proto: optional prototype context [batch, hid_dim] or [1, hid_dim]
            proto_alpha: prototype blend weight

        Returns:
            loss: scalar training loss
            score: per-node anomaly score [batch]
            reconstructed: sampled reconstruction [batch, hid_dim]
        """
        # Conditional vs unconditional FM loss
        if proto is not None:
            graph_context = proto
            # logit_normal t-sampling emphasizes boundary times
            loss = conditional_flow_matching_loss(
                self.velocity_fn,
                x_1,
                graph_context,
                t_sampling='logit_normal',
                reduction='mean'
            )
        else:
            graph_context = torch.zeros(1, x_1.shape[1], device=x_1.device)
            loss = flow_matching_loss(
                self.velocity_fn,
                x_1,
                graph_context,
                reduction='mean'
            )

        # Auxiliary path: short FM sampling for reconstruction error as score
        with torch.no_grad():
            x_0 = torch.randn_like(x_1)
            reconstructed = sample_flow_matching(
                self.velocity_fn, x_0, num_steps=10,
                proto=proto, proto_alpha=proto_alpha
            )

        score = torch.sqrt(torch.sum((x_1 - reconstructed) ** 2, dim=1))

        return loss, score, reconstructed


def sample_flow_matching(velocity_net, x_0, num_steps=50, proto=None, proto_alpha=None):
    """
    Euler ODE sampler for flow matching: integrate forward from noise x_0 toward data.

    Args:
        velocity_net: velocity field
        x_0: initial noise [batch, dim]
        num_steps: Euler steps
        proto: optional prototype context
        proto_alpha: prototype weight
    Returns:
        x_1: generated state [batch, dim]
    """
    device = x_0.device
    batch_size = x_0.shape[0]

    # Forward time grid 0 -> 1
    dt = 1.0 / num_steps
    x_t = x_0.clone()

    velocity_net.eval()
    with torch.no_grad():
        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=device, dtype=x_0.dtype)

            v_t = velocity_net(x_t, t, context=proto, proto_alpha=proto_alpha)

            # Euler: x <- x + dt * v
            x_t = x_t + dt * v_t

    return x_t


def sample_flow_matching_free(proto_net, free_net, x_0, num_steps=50, proto=None, proto_alpha=None, weight=None):
    """
    Combine prototype-conditioned and free velocity nets during Euler sampling.

    Args:
        proto_net: conditional velocity (uses prototype)
        free_net: unconditional velocity
        x_0: initial noise [batch, dim]
        num_steps: Euler steps
        proto: prototype tensor
        proto_alpha: prototype strength
        weight: blend weight; larger weight pushes toward the proto branch
    Returns:
        x_1: generated state [batch, dim]
    """
    device = x_0.device
    batch_size = x_0.shape[0]

    dt = 1.0 / num_steps
    x_t = x_0.clone()

    proto_net.eval()
    free_net.eval()

    with torch.no_grad():
        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=device, dtype=x_0.dtype)

            v_proto = proto_net(x_t, t, context=proto, proto_alpha=proto_alpha)
            v_free = free_net(x_t, t, context=None, proto_alpha=None)

            v_combined = (1.0 + weight) * v_free - weight * v_proto

            x_t = x_t + dt * v_combined

    return x_t
