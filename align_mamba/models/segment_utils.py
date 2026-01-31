"""Segment processing utilities for packed sequences."""

from typing import List, Tuple, Optional

import torch
import torch.nn as nn


def _segment_boundaries(cu_seqlens: torch.Tensor) -> List[Tuple[int, int]]:
    """Convert cumulative lengths to (start, end) pairs."""
    b = cu_seqlens.tolist()
    return [(b[i], b[i + 1]) for i in range(len(b) - 1)]


def process_segments(
    x: torch.Tensor,
    mamba_fwd: nn.Module,
    cu_seqlens: torch.Tensor,
    mamba_bwd: Optional[nn.Module] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Process segments with Mamba (unidirectional or bidirectional).

    Args:
        x: Packed input tensor (total_tokens, d_model)
        mamba_fwd: Forward Mamba module
        cu_seqlens: Cumulative sequence lengths
        mamba_bwd: Backward Mamba module (None for unidirectional)

    Returns:
        (forward_output, backward_output) where backward_output is None if unidirectional
    """
    fwd_outputs = []
    bwd_outputs = [] if mamba_bwd else None

    for start, end in _segment_boundaries(cu_seqlens):
        seq = x[start:end].unsqueeze(0)
        fwd_outputs.append(mamba_fwd(seq).squeeze(0))

        if mamba_bwd:
            seq_flipped = torch.flip(seq, dims=[1])
            y_bwd = torch.flip(mamba_bwd(seq_flipped), dims=[1])
            bwd_outputs.append(y_bwd.squeeze(0))

    fwd_out = torch.cat(fwd_outputs, dim=0)
    bwd_out = torch.cat(bwd_outputs, dim=0) if bwd_outputs else None
    return fwd_out, bwd_out
