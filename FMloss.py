import torch
import torch.nn.functional as F
import numpy as np

def flow_matching_loss(model, x_1, graph_context, reduction='mean', weight=None):
    """
    Basic flow-matching MSE between predicted and target velocity on a linear path.

    Args:
        model: velocity net, inputs (x_t, t, context) -> v
        x_1: data (Batch, Dim)
        graph_context: structural context (Batch, Dim) or (1, Dim) broadcastable
        reduction: 'mean' or 'none'
        weight: optional per-sample weights (Batch,)

    Returns:
        loss: scalar if reduction='mean' else per-sample vector
    """
    batch_size = x_1.shape[0]

    # 1) sample t ~ U[0,1]
    t = torch.rand(batch_size, 1, device=x_1.device)

    # 2) sample Gaussian noise x_0
    x_0 = torch.randn_like(x_1)

    # 3) linear bridge x_t = (1-t)*x_0 + t*x_1
    x_t = (1 - t) * x_0 + t * x_1

    # 4) target velocity along the straight path
    target_v = x_1 - x_0

    # 5) broadcast context if needed
    if graph_context.dim() == 2:
        if graph_context.shape[0] == 1 and batch_size > 1:
            graph_context = graph_context.repeat(batch_size, 1)
        elif graph_context.shape[0] != batch_size:
            raise ValueError(f"Context shape mismatch: {graph_context.shape[0]} vs batch_size {batch_size}")

    pred_v = model(x_t, t.squeeze(-1) if t.dim() > 1 else t, graph_context)

    mse_per_sample = F.mse_loss(pred_v, target_v, reduction='none').mean(dim=1)

    if weight is not None:
        mse_per_sample = mse_per_sample * weight

    # reduction
    if reduction == 'mean':
        return mse_per_sample.mean()
    elif reduction == 'none':
        return mse_per_sample
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def conditional_flow_matching_loss(model, x_1, graph_context, t_sampling='uniform',
                                    reduction='mean', weight=None):
    """
    Conditional flow-matching loss with optional non-uniform time sampling.

    Args:
        model: velocity net
        x_1: data (Batch, Dim)
        graph_context: context (Batch, Dim) or (1, Dim)
        t_sampling: 'uniform' or 'logit_normal'
        reduction: 'mean' or 'none'
        weight: optional per-sample weights

    Returns:
        loss: scalar or vector per reduction
    """
    batch_size = x_1.shape[0]

    if t_sampling == 'uniform':
        t = torch.rand(batch_size, 1, device=x_1.device)
    elif t_sampling == 'logit_normal':
        # emphasize boundary times
        t_normal = torch.randn(batch_size, 1, device=x_1.device)
        t = torch.sigmoid(t_normal * 2.0)
    else:
        raise ValueError(f"Unknown t_sampling: {t_sampling}")

    x_0 = torch.randn_like(x_1)

    x_t = (1 - t) * x_0 + t * x_1

    target_v = x_1 - x_0

    if graph_context.dim() == 2:
        if graph_context.shape[0] == 1 and batch_size > 1:
            graph_context = graph_context.repeat(batch_size, 1)

    t_input = t.squeeze(-1) if t.dim() > 1 else t
    pred_v = model(x_t, t_input, graph_context)

    mse_per_sample = F.mse_loss(pred_v, target_v, reduction='none').mean(dim=1)

    if weight is not None:
        mse_per_sample = mse_per_sample * weight

    if reduction == 'mean':
        return mse_per_sample.mean()
    elif reduction == 'none':
        return mse_per_sample
    else:
        raise ValueError(f"Unknown reduction: {reduction}")
