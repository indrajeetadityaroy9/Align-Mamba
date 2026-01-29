"""Shared utilities for Mamba models with optimized segment processing."""

from typing import Optional, List, Tuple
import torch
import torch.nn as nn


def compute_segment_boundaries(
    cu_seqlens: torch.Tensor,
) -> List[Tuple[int, int]]:
    """
    Pre-compute segment boundaries on CPU once, avoiding repeated GPU->CPU syncs.

    Args:
        cu_seqlens: Cumulative sequence lengths (batch_size + 1,)

    Returns:
        List of (start, end) tuples for each segment
    """
    # Single GPU->CPU transfer for all boundaries
    boundaries = cu_seqlens.tolist()
    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


def process_segments_bimamba(
    x: torch.Tensor,
    mamba_fwd: nn.Module,
    mamba_bwd: nn.Module,
    cu_seqlens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Process segments with bidirectional Mamba, optimized to minimize GPU-CPU syncs.

    Uses a single .tolist() call instead of per-segment .item() calls.

    Args:
        x: Input tensor (total_tokens, d_model)
        mamba_fwd: Forward Mamba module
        mamba_bwd: Backward Mamba module
        cu_seqlens: Cumulative sequence lengths

    Returns:
        Tuple of (forward_outputs, backward_outputs)
    """
    # Single GPU->CPU sync for all boundaries
    boundaries = compute_segment_boundaries(cu_seqlens)

    fwd_outputs = []
    bwd_outputs = []

    for start, end in boundaries:
        seq = x[start:end].unsqueeze(0)  # (1, seq_len, d_model)

        # Forward pass
        y_fwd = mamba_fwd(seq)
        fwd_outputs.append(y_fwd.squeeze(0))

        # Backward pass (flip -> process -> flip back)
        seq_flipped = torch.flip(seq, dims=[1])
        y_bwd_rev = mamba_bwd(seq_flipped)
        y_bwd = torch.flip(y_bwd_rev, dims=[1])
        bwd_outputs.append(y_bwd.squeeze(0))

    return torch.cat(fwd_outputs, dim=0), torch.cat(bwd_outputs, dim=0)


def process_segments_unidirectional(
    x: torch.Tensor,
    mamba: nn.Module,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """
    Process segments with unidirectional Mamba, optimized to minimize GPU-CPU syncs.

    Uses a single .tolist() call instead of per-segment .item() calls.

    Args:
        x: Input tensor (total_tokens, d_model)
        mamba: Mamba module
        cu_seqlens: Cumulative sequence lengths

    Returns:
        Processed output tensor
    """
    # Single GPU->CPU sync for all boundaries
    boundaries = compute_segment_boundaries(cu_seqlens)

    outputs = []
    for start, end in boundaries:
        seq = x[start:end].unsqueeze(0)  # (1, seq_len, d_model)
        out = mamba(seq)
        outputs.append(out.squeeze(0))

    return torch.cat(outputs, dim=0)


def segment_aware_flip_optimized(
    x: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Flip sequences respecting document boundaries, optimized.

    Args:
        x: Input tensor (total_tokens, ...) or (batch, seq_len, ...)
        cu_seqlens: Cumulative sequence lengths for packed sequences

    Returns:
        Flipped tensor
    """
    if cu_seqlens is None:
        return torch.flip(x, dims=[1])

    # Single GPU->CPU sync for all boundaries
    boundaries = compute_segment_boundaries(cu_seqlens)

    flipped_segments = []
    for start, end in boundaries:
        segment = x[start:end]
        flipped_segments.append(torch.flip(segment, dims=[0]))

    return torch.cat(flipped_segments, dim=0)


def get_unwrapped_model(model: nn.Module) -> nn.Module:
    """
    Unwrap model from torch.compile and DDP/FSDP wrappers.

    Handles:
    - torch.compile: _orig_mod attribute
    - DDP/FSDP: module attribute

    Args:
        model: Potentially wrapped model

    Returns:
        Unwrapped model
    """
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    if hasattr(model, "module"):
        model = model.module
    return model


# Weight decay parameter patterns - centralized for consistency
NO_DECAY_PATTERNS = [
    "bias",
    "LayerNorm.weight",
    "layer_norm.weight",
    "norm.weight",
    "RMSNorm.weight",
    "rmsnorm.weight",
]


def split_params_for_weight_decay(
    model: nn.Module,
    weight_decay: float,
    patterns: Optional[List[str]] = None,
) -> List[dict]:
    """
    Split model parameters into groups with and without weight decay.

    Args:
        model: Model to get parameters from
        weight_decay: Weight decay value for decayed parameters
        patterns: Patterns for parameters that should NOT have weight decay
                  Defaults to NO_DECAY_PATTERNS

    Returns:
        List of parameter groups for optimizer
    """
    if patterns is None:
        patterns = NO_DECAY_PATTERNS

    params_with_wd = []
    params_without_wd = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(pattern in name for pattern in patterns):
            params_without_wd.append(param)
        else:
            params_with_wd.append(param)

    return [
        {"params": params_with_wd, "weight_decay": weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]


__all__ = [
    "compute_segment_boundaries",
    "process_segments_bimamba",
    "process_segments_unidirectional",
    "segment_aware_flip_optimized",
    "get_unwrapped_model",
    "split_params_for_weight_decay",
    "NO_DECAY_PATTERNS",
]
