"""MemMamba: Cross-layer memory pool for long-range retrieval (arXiv:2510.03279)."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba2

from align_mamba.models.components import RMSNorm
from align_mamba.models.registry import BlockRegistry


class TokenImportanceScorer(nn.Module):
    """MLP-based importance scorer for memory insertion."""

    def __init__(
        self,
        d_model: int,
        hidden_ratio: float = 0.25,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        hidden_dim = int(d_model * hidden_ratio)
        factory_kwargs = {"device": device, "dtype": dtype}

        self.w1 = nn.Linear(d_model, hidden_dim, bias=False, **factory_kwargs)
        self.w2 = nn.Linear(hidden_dim, 1, bias=False, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, T, d_model)

        Returns:
            Importance scores (B, T)
        """
        h = F.relu(self.w1(x))
        scores = torch.sigmoid(self.w2(h)).squeeze(-1)
        return scores


class CrossTokenRetrieval(nn.Module):
    """Attention-based memory retrieval."""

    def __init__(
        self,
        d_model: int,
        summary_dim: int = 64,
        n_heads: int = 4,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.summary_dim = summary_dim
        self.n_heads = n_heads
        self.head_dim = summary_dim // n_heads

        factory_kwargs = {"device": device, "dtype": dtype}

        # Query projection (from current features)
        self.q_proj = nn.Linear(d_model, summary_dim, bias=False, **factory_kwargs)

        # Key/Value projections (from memory summaries)
        self.k_proj = nn.Linear(summary_dim, summary_dim, bias=False, **factory_kwargs)
        self.v_proj = nn.Linear(summary_dim, d_model, bias=False, **factory_kwargs)

    def forward(
        self,
        x: torch.Tensor,
        memory_pool: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Query features (B, T, d_model)
            memory_pool: Memory pool (B, pool_size, summary_dim)
            memory_mask: Valid memory mask (B, pool_size) - True for valid entries

        Returns:
            Retrieved features (B, T, d_model)
        """
        B, T, _ = x.shape
        pool_size = memory_pool.size(1)

        # Check if any batch has valid memories
        if not memory_mask.any():
            return torch.zeros_like(x)

        # Compute attention
        q = self.q_proj(x)  # (B, T, summary_dim)
        k = self.k_proj(memory_pool)  # (B, pool_size, summary_dim)
        v = self.v_proj(memory_pool)  # (B, pool_size, d_model)

        # Attention scores
        scale = 1.0 / (self.head_dim ** 0.5)
        attn = torch.bmm(q, k.transpose(1, 2)) * scale  # (B, T, pool_size)

        # Mask invalid memory entries
        attn_mask = ~memory_mask.unsqueeze(1).expand(-1, T, -1)  # (B, T, pool_size)
        attn = attn.masked_fill(attn_mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)  # Handle all-masked rows

        # Retrieve
        out = torch.bmm(attn, v)  # (B, T, d_model)

        return out


@BlockRegistry.register("memmamba")
class MemMambaBlock(nn.Module):
    """Mamba with cross-layer memory pool (arXiv:2510.03279)."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        pool_size: int = 50,
        summary_dim: int = 64,
        tau1: float = 0.5,
        tau2: float = 0.3,
        layer_idx: int = 0,
        cross_layer_frequency: int = 4,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs,  # Accept extra kwargs for registry compatibility
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.pool_size = pool_size
        self.summary_dim = summary_dim
        self.tau1 = tau1  # Threshold for memory insertion
        self.tau2 = tau2  # Threshold for memory retrieval
        self.layer_idx = layer_idx
        self.cross_layer_frequency = cross_layer_frequency

        factory_kwargs = {"device": device, "dtype": dtype}

        self.norm = RMSNorm(d_model, device=device, dtype=dtype)

        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            **factory_kwargs,
        )

        # Memory components
        self.scorer = TokenImportanceScorer(d_model, **factory_kwargs)
        self.summarizer = nn.Linear(d_model, summary_dim, bias=False, **factory_kwargs)
        self.retrieval = CrossTokenRetrieval(d_model, summary_dim, **factory_kwargs)

        # Fusion gate for combining Mamba output with retrieved memory
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model, bias=False, **factory_kwargs),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        inference_params: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        memory_state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, T, d_model)
            inference_params: Optional SSM state for autoregressive decoding
            cu_seqlens: Optional cumulative sequence lengths for packed sequences
            memory_state: Optional tuple of (pool, priorities, counts) for cross-layer memory
                         Each is (B, pool_size, summary_dim), (B, pool_size), (B,)

        Returns:
            Output tensor (B, T, d_model)
        """
        residual = x
        x = self.norm(x)
        x = x.contiguous()

        B, T, D = x.shape

        # Mamba processing
        if inference_params is not None:
            conv_state, ssm_state = inference_params
            y, conv_state_out, ssm_state_out = self.mamba.step(x, conv_state, ssm_state)
            conv_state.copy_(conv_state_out)
            ssm_state.copy_(ssm_state_out)
        else:
            y = self.mamba(x)

        # Score token importance
        scores = self.scorer(y)  # (B, T)

        # Initialize or use provided memory state
        if memory_state is None:
            pool = torch.zeros(B, self.pool_size, self.summary_dim, device=x.device, dtype=x.dtype)
            priorities = torch.zeros(B, self.pool_size, device=x.device, dtype=x.dtype)
            counts = torch.zeros(B, device=x.device, dtype=torch.long)
        else:
            pool, priorities, counts = memory_state

        # Update memory pool at cross-layer positions
        if self.layer_idx % self.cross_layer_frequency == 0:
            pool, priorities, counts = self._update_memory_batch(
                y, scores, pool, priorities, counts
            )

        # Retrieve from memory if average importance is high
        mean_score = scores.mean(dim=1)  # (B,)
        retrieve_mask = (mean_score > self.tau2) & (counts > 0)

        if retrieve_mask.any():
            # Create memory validity mask
            memory_mask = torch.arange(self.pool_size, device=x.device).unsqueeze(0) < counts.unsqueeze(1)

            retrieved = self.retrieval(y, pool, memory_mask)

            # Gated fusion (only for batches that should retrieve)
            gate_input = torch.cat([y, retrieved], dim=-1)
            gate = self.fusion_gate(gate_input)

            # Apply retrieval only where retrieve_mask is True
            y = y + gate * retrieved * retrieve_mask.view(B, 1, 1)

        return residual + y

    def _update_memory_batch(
        self,
        tokens: torch.Tensor,
        importance: torch.Tensor,
        pool: torch.Tensor,
        priorities: torch.Tensor,
        counts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Priority-based memory update (vectorized, no GPU→CPU sync)."""
        B, T, D = tokens.shape

        # Mask for important tokens (above threshold)
        important_mask = importance > self.tau1  # (B, T)

        # Sort tokens by importance (descending) within each batch
        sorted_importance, sorted_indices = torch.sort(importance, dim=1, descending=True)

        # Gather tokens in sorted order
        sorted_tokens = torch.gather(
            tokens, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, D)
        )  # (B, T, D)

        # Gather mask in sorted order
        sorted_mask = torch.gather(important_mask, 1, sorted_indices)  # (B, T)

        # Batch-summarize all sorted tokens at once (more efficient than one-by-one)
        sorted_summaries = self.summarizer(sorted_tokens.view(B * T, D)).view(B, T, -1)

        # For each batch, process important tokens up to pool_size
        # We'll process one slot at a time to handle priority-based replacement
        for slot in range(min(T, self.pool_size)):
            # Check if this position has an important token
            has_important = sorted_mask[:, slot]  # (B,)
            slot_importance = sorted_importance[:, slot]  # (B,)
            slot_summary = sorted_summaries[:, slot]  # (B, summary_dim)

            # Case 1: Pool not full - add directly
            not_full = counts < self.pool_size
            add_mask = has_important & not_full  # (B,)

            if add_mask.any():
                # Use counts as insertion index
                insert_idx = counts.clone()  # (B,)
                # Expand for scatter
                insert_idx_expanded = insert_idx.unsqueeze(-1).expand(-1, slot_summary.size(-1))

                # Only update for batches where add_mask is True
                for b in range(B):
                    if add_mask[b]:
                        idx = insert_idx[b]
                        pool[b, idx] = slot_summary[b]
                        priorities[b, idx] = slot_importance[b]
                        counts[b] += 1

            # Case 2: Pool full - replace lowest priority if better
            is_full = counts >= self.pool_size
            replace_candidate = has_important & is_full  # (B,)

            if replace_candidate.any():
                # Find minimum priority in each batch's pool
                min_priorities, min_indices = priorities.min(dim=1)  # (B,), (B,)

                # Only replace if new score > min priority
                should_replace = replace_candidate & (slot_importance > min_priorities)

                for b in range(B):
                    if should_replace[b]:
                        idx = min_indices[b]
                        pool[b, idx] = slot_summary[b]
                        priorities[b, idx] = slot_importance[b]

        return pool, priorities, counts

    def allocate_inference_cache(
        self,
        batch_size: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Allocate state cache for autoregressive decoding.

        Returns:
            Tuple of (conv_state, ssm_state) for Mamba
        """
        d_inner = self.d_model * self.expand
        dtype = dtype or torch.bfloat16
        device = device or next(self.parameters()).device

        conv_state = torch.zeros(batch_size, d_inner, self.d_conv, dtype=dtype, device=device)
        ssm_state = torch.zeros(batch_size, d_inner, self.d_state, dtype=dtype, device=device)

        return conv_state, ssm_state
