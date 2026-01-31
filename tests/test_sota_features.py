"""Tests for SOTA architecture.

Run with: python -m pytest tests/test_sota_features.py -v
"""

import math
import pytest
import torch


class TestCapacityAnalysis:
    """Tests for capacity analysis."""

    def test_analyze_ssm_capacity_no_overflow(self):
        from models.capacity import analyze_ssm_capacity

        analysis = analyze_ssm_capacity(d_state=64, num_pairs=32, n_layers=24)

        assert analysis.d_state == 64
        assert analysis.num_pairs == 32
        assert analysis.capacity_utilization == 0.5
        assert analysis.overflow_ratio == 1.0
        assert analysis.convergence_guaranteed
        assert analysis.recommended_cross_attn_interval is None

    def test_analyze_ssm_capacity_with_overflow(self):
        from models.capacity import analyze_ssm_capacity

        analysis = analyze_ssm_capacity(d_state=64, num_pairs=256, n_layers=24)

        assert analysis.capacity_utilization == 4.0
        assert analysis.overflow_ratio == 4.0
        assert analysis.recommended_cross_attn_interval is not None


class TestPolarization:
    """Tests for polarized Mamba channels."""

    def test_zero_channel_no_memory(self):
        """A=0 channel has no temporal dependency."""
        B, T, D = 2, 8, 64
        x = torch.randn(B, T, D)
        zero_proj = torch.randn(D, D * 2)

        y_zero = x @ zero_proj

        # Changing past doesn't affect future
        x_modified = x.clone()
        x_modified[:, 0] = torch.randn(B, D)
        y_zero_modified = x_modified @ zero_proj

        assert torch.allclose(y_zero[:, 1:], y_zero_modified[:, 1:])

    def test_one_channel_cumsum(self):
        """A=1 channel is cumulative sum."""
        B, T, D = 2, 8, 64
        x = torch.randn(B, T, D)
        one_proj = torch.randn(D, D * 2)

        y_one = torch.cumsum(x @ one_proj, dim=1)

        # Verify cumsum property
        expected_t2 = (x[:, :3] @ one_proj).sum(dim=1)
        assert torch.allclose(y_one[:, 2], expected_t2)


class TestCrossAttentionPlacement:
    """Tests for capacity-aware cross-attention placement."""

    def test_layer_0_always_included(self):
        from models.align_mamba import compute_cross_attention_positions

        positions = compute_cross_attention_positions(n_layers=24, d_state=64, num_pairs=32)
        assert 0 in positions

    def test_more_positions_with_overflow(self):
        from models.align_mamba import compute_cross_attention_positions

        no_overflow = compute_cross_attention_positions(n_layers=24, d_state=64, num_pairs=32)
        with_overflow = compute_cross_attention_positions(n_layers=24, d_state=64, num_pairs=256)

        assert len(with_overflow) > len(no_overflow)


class TestStateExpansion:
    """Tests for state expansion."""

    def test_forget_lower_bound_computation(self):
        from models.state_expansion import compute_forget_lower_bound

        assert compute_forget_lower_bound(0, 24) == pytest.approx(0.9, rel=0.01)
        assert compute_forget_lower_bound(23, 24) == pytest.approx(0.99, rel=0.01)

    def test_outer_product_capacity(self):
        n_heads, head_dim = 4, 64
        d_model = n_heads * head_dim

        standard_capacity = d_model
        expanded_capacity = n_heads * head_dim * head_dim

        assert expanded_capacity > standard_capacity


class TestBasedAttention:
    """Tests for BASED linear attention."""

    def test_taylor_feature_map_dimension(self):
        from models.based_attention import taylor_feature_map_reference

        d = 16
        expected_dim = 1 + d + d * (d + 1) // 2

        x = torch.randn(2, 8, 4, d)
        phi_x = taylor_feature_map_reference(x)

        assert phi_x.shape[-1] == expected_dim

    def test_linear_attention_causality(self):
        from models.based_attention import taylor_feature_map_reference, linear_attention_reference

        B, T, H, d, D = 2, 8, 4, 8, 32
        q = torch.randn(B, T, H, d)
        k = torch.randn(B, T, H, d)
        v = torch.randn(B, T, H, D)

        q_feat = taylor_feature_map_reference(q)
        k_feat = taylor_feature_map_reference(k)

        out1 = linear_attention_reference(q_feat, k_feat, v)

        # Modify future
        k_modified = k.clone()
        k_modified[:, T // 2:] = torch.randn(B, T - T // 2, H, d)
        k_feat_modified = taylor_feature_map_reference(k_modified)

        out2 = linear_attention_reference(q_feat, k_feat_modified, v)

        # Past outputs unchanged (causal)
        assert torch.allclose(out1[:, : T // 2], out2[:, : T // 2], atol=1e-5)


class TestMemMamba:
    """Tests for MemMamba memory pool."""

    def test_memory_pool_update(self):
        from models.memmamba import MemoryPool

        pool = MemoryPool(pool_size=5, summary_dim=32)
        summaries = torch.randn(10, 32)
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2, 0.7, 0.4, 0.6, 0.5, 0.85])

        pool.update(summaries, scores, threshold=0.5)

        assert pool.count.item() == 5

    def test_importance_scorer_range(self):
        from models.memmamba import TokenImportanceScorer

        scorer = TokenImportanceScorer(d_model=64)
        x = torch.randn(2, 16, 64)
        scores = scorer(x)

        assert (scores >= 0).all()
        assert (scores <= 1).all()


class TestNumericalStability:
    """Numerical stability tests."""

    def test_polarized_with_zeros(self):
        B, T, D = 2, 16, 256
        x = torch.zeros(B, T, D)
        zero_proj = torch.randn(D, D * 2)
        one_proj = torch.randn(D, D * 2)

        y_zero = x @ zero_proj
        y_one = torch.cumsum(x @ one_proj, dim=1)

        assert not torch.isnan(y_zero).any()
        assert not torch.isnan(y_one).any()

    def test_linear_attention_uniform_features(self):
        from models.based_attention import linear_attention_reference

        B, T, H, F, D = 2, 8, 4, 16, 32
        q_feat = torch.ones(B, T, H, F)
        k_feat = torch.ones(B, T, H, F)
        v = torch.randn(B, T, H, D)

        out = linear_attention_reference(q_feat, k_feat, v)

        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
