"""Attention mechanisms for Hybrid Mamba-Attention architecture."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from flash_attn import flash_attn_func, flash_attn_varlen_func

from .components import RMSNorm, RotaryPositionalEmbedding
from .registry import AttentionRegistry


def broadcast_mask_for_sdpa(
    mask: torch.Tensor,
    batch_size: int,
    dtype: torch.dtype,
    src_len: int = None,
) -> torch.Tensor:
    """Convert padding mask to SDPA-compatible attention mask.

    Transforms a (batch, seq_len) boolean/float mask into the format expected
    by scaled_dot_product_attention: (batch, 1, 1, seq_len) with -inf for masked positions.
    """
    attn_mask = mask.unsqueeze(1).unsqueeze(2)
    assert attn_mask.dim() == 4
    assert attn_mask.shape[0] == batch_size
    if src_len is not None:
        assert attn_mask.shape[-1] == src_len
    attn_mask = attn_mask.to(dtype=dtype)
    return (1.0 - attn_mask) * torch.finfo(dtype).min


class BidirectionalAttention(nn.Module):
    """Bidirectional (non-causal) attention for encoder sparse layers."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.0,
        max_seq_len: int = 8192,
        bias: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        factory_kwargs = {"device": device, "dtype": dtype}

        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)
        self.rope = RotaryPositionalEmbedding(dim=self.head_dim, max_seq_len=max_seq_len, device=device)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> torch.Tensor:
        is_packed = cu_seqlens is not None

        if is_packed:
            residual = x
            x = self.norm(x)
            total_tokens = x.size(0)

            qkv = self.qkv_proj(x)
            qkv = qkv.view(total_tokens, 3, self.n_heads, self.head_dim)
            q, k, v = qkv.unbind(dim=1)

            out = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=self.dropout if self.training else 0.0,
                causal=False,
            )

            out = out.view(total_tokens, self.d_model)
            out = self.out_proj(out)
            return residual + out
        else:
            residual = x
            x = self.norm(x)
            B, T, _ = x.shape

            qkv = self.qkv_proj(x)
            qkv = qkv.view(B, T, 3, self.n_heads, self.head_dim)
            q, k, v = qkv.unbind(dim=2)

            q, k = self.rope.apply_to_qk_bthd(q, k, q_offset=0, k_offset=0)

            if attention_mask is not None:
                attn_mask = broadcast_mask_for_sdpa(attention_mask, B, q.dtype)

                q_sdpa = q.transpose(1, 2)
                k_sdpa = k.transpose(1, 2)
                v_sdpa = v.transpose(1, 2)
                out = F.scaled_dot_product_attention(
                    q_sdpa, k_sdpa, v_sdpa,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=False,
                )
                out = out.transpose(1, 2)
            else:
                out = flash_attn_func(q, k, v, dropout_p=self.dropout if self.training else 0.0, causal=False)

            out = out.view(B, T, self.d_model)
            out = self.out_proj(out)
            return residual + out


@AttentionRegistry.register("softmax")
class FlashCrossAttention(nn.Module):
    """Cross-attention with FlashAttention-2 and QK-Norm for stability."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.0,
        max_seq_len: int = 8192,
        bias: bool = False,
        use_rope: bool = True,
        use_qk_norm: bool = True,  # QK-Norm for cross-attention stability
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs,  # Accept extra kwargs for registry compatibility
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.use_rope = use_rope
        self.use_qk_norm = use_qk_norm

        factory_kwargs = {"device": device, "dtype": dtype}

        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.q_proj = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)
        self.kv_proj = nn.Linear(d_model, d_model * 2, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)

        if use_rope:
            self.rope = RotaryPositionalEmbedding(dim=self.head_dim, max_seq_len=max_seq_len, device=device)
        else:
            self.rope = None

        # Softmax scale for attention (sqrt(head_dim) is standard)
        self.softmax_scale = math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        decoder_offset: int = 0,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
    ) -> torch.Tensor:
        is_packed = cu_seqlens_q is not None

        if is_packed:
            residual = x
            x = self.norm(x)
            total_dec_tokens = x.size(0)
            total_enc_tokens = encoder_out.size(0)

            q = self.q_proj(x)
            q = q.view(total_dec_tokens, self.n_heads, self.head_dim)

            kv = self.kv_proj(encoder_out)
            kv = kv.view(total_enc_tokens, 2, self.n_heads, self.head_dim)
            k, v = kv.unbind(dim=1)

            if self.use_qk_norm:
                q = F.normalize(q, p=2, dim=-1)
                k = F.normalize(k, p=2, dim=-1)

            out = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                dropout_p=self.dropout if self.training else 0.0,
                softmax_scale=self.softmax_scale,
                causal=False,
            )

            out = out.view(total_dec_tokens, self.d_model)
            out = self.out_proj(out)
            return residual + out
        else:
            residual = x
            x = self.norm(x)
            B, T_dec, _ = x.shape
            _, T_enc, _ = encoder_out.shape

            q = self.q_proj(x)
            q = q.view(B, T_dec, self.n_heads, self.head_dim)

            kv = self.kv_proj(encoder_out)
            kv = kv.view(B, T_enc, 2, self.n_heads, self.head_dim)
            k, v = kv.unbind(dim=2)

            if self.rope is not None:
                q, k = self.rope.apply_to_qk_bthd(q, k, q_offset=decoder_offset, k_offset=0)

            if self.use_qk_norm:
                q = F.normalize(q, p=2, dim=-1)
                k = F.normalize(k, p=2, dim=-1)

            if encoder_padding_mask is not None:
                attn_mask = broadcast_mask_for_sdpa(encoder_padding_mask, B, q.dtype, src_len=T_enc)

                q_sdpa = q.transpose(1, 2)
                k_sdpa = k.transpose(1, 2)
                v_sdpa = v.transpose(1, 2)
                out = F.scaled_dot_product_attention(
                    q_sdpa, k_sdpa, v_sdpa,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout if self.training else 0.0,
                    scale=self.softmax_scale,
                    is_causal=False,
                )
                out = out.transpose(1, 2)
            else:
                out = flash_attn_func(
                    q, k, v,
                    dropout_p=self.dropout if self.training else 0.0,
                    softmax_scale=self.softmax_scale,
                    causal=False,
                )

            out = out.view(B, T_dec, self.d_model)
            out = self.out_proj(out)
            return residual + out


def taylor_feature_map(x: torch.Tensor) -> torch.Tensor:
    """2nd-order Taylor exp approximation for linear attention (arXiv:2402.18668)."""
    B, T, H, d = x.shape
    ones = torch.ones(B, T, H, 1, dtype=x.dtype, device=x.device)
    outer = torch.einsum('bthd,bthe->bthde', x, x)
    scale = torch.full((d, d), 1.0 / math.sqrt(2), dtype=x.dtype, device=x.device)
    scale.fill_diagonal_(0.5)
    outer = outer * scale
    triu_i, triu_j = torch.triu_indices(d, d, device=x.device)
    quadratic = outer[..., triu_i, triu_j]
    return torch.cat([ones, x, quadratic], dim=-1)


@AttentionRegistry.register("based")
class BasedCrossAttention(nn.Module):
    """Taylor linear attention + sliding window (arXiv:2402.18668)."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        feature_dim: int = 16,
        window_size: int = 64,
        dropout: float = 0.0,
        max_seq_len: int = 8192,
        bias: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs,  # Accept extra kwargs for registry compatibility
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.feature_dim = feature_dim
        self.window_size = window_size
        self.dropout = dropout

        self.expanded_dim = 1 + feature_dim + feature_dim * (feature_dim + 1) // 2

        factory_kwargs = {"device": device, "dtype": dtype}

        self.norm = RMSNorm(d_model, device=device, dtype=dtype)

        self.q_proj = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)
        self.kv_proj = nn.Linear(d_model, d_model * 2, bias=bias, **factory_kwargs)
        self.q_feature = nn.Linear(self.head_dim, feature_dim, bias=False, **factory_kwargs)
        self.k_feature = nn.Linear(self.head_dim, feature_dim, bias=False, **factory_kwargs)
        self.window_kv = nn.Linear(d_model, d_model * 2, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(d_model * 2, d_model, bias=bias, **factory_kwargs)

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        decoder_offset: int = 0,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
    ) -> torch.Tensor:
        if cu_seqlens_q is not None:
            raise NotImplementedError("BASED attention does not support packed sequences")

        residual = x
        x = self.norm(x)
        B, T_dec, _ = x.shape
        _, T_enc, _ = encoder_out.shape

        q = self.q_proj(x).view(B, T_dec, self.n_heads, self.head_dim)
        kv = self.kv_proj(encoder_out)
        kv = kv.view(B, T_enc, 2, self.n_heads, self.head_dim)
        k, v = kv.unbind(dim=2)

        q_feat = self.q_feature(q)
        k_feat = self.k_feature(k)
        q_phi = taylor_feature_map(q_feat)
        k_phi = taylor_feature_map(k_feat)

        kv_state = torch.einsum('bshf,bshd->bhfd', k_phi, v)
        k_state = k_phi.sum(dim=1)

        if encoder_padding_mask is not None:
            mask = encoder_padding_mask.unsqueeze(-1).unsqueeze(-1)
            k_phi_masked = k_phi * mask
            v_masked = v * mask.squeeze(-1)
            kv_state = torch.einsum('bshf,bshd->bhfd', k_phi_masked, v_masked)
            k_state = k_phi_masked.sum(dim=1)

        linear_out = torch.einsum('bthf,bhfd->bthd', q_phi, kv_state)
        normalizer = torch.einsum('bthf,bhf->bth', q_phi, k_state).unsqueeze(-1)
        linear_out = linear_out / (normalizer + 1e-6)
        linear_out = linear_out.view(B, T_dec, self.d_model)

        kv_win = self.window_kv(x).view(B, T_dec, 2, self.n_heads, self.head_dim)
        k_win, v_win = kv_win.unbind(dim=2)
        q_win = q

        window_out = flash_attn_func(
            q_win, k_win, v_win,
            dropout_p=self.dropout if self.training else 0.0,
            causal=True,
            window_size=(self.window_size, 0),
        )
        window_out = window_out.view(B, T_dec, self.d_model)

        combined = torch.cat([linear_out, window_out], dim=-1)
        out = self.out_proj(combined)

        if self.dropout > 0 and self.training:
            out = F.dropout(out, p=self.dropout, training=True)

        return residual + out
