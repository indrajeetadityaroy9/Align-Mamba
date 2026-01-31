"""Shared utilities for Mamba models with optimized segment processing."""

from typing import List, Tuple
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


__all__ = [
    "compute_segment_boundaries",
    "process_segments_bimamba",
    "process_segments_unidirectional",
    "get_unwrapped_model",
]
