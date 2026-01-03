"""
Unit tests for the Hybrid Mamba-Attention model architecture.

Run with: python -m pytest tests/test_models.py -v
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Check if mamba-ssm is available
try:
    import mamba_ssm
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False

requires_mamba = pytest.mark.skipif(not HAS_MAMBA, reason="mamba-ssm required (CUDA only)")

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32


class TestImports:
    """Test that all model components can be imported."""

    def test_layers_imports(self):
        from models.layers import RMSNorm, RotaryPositionalEmbedding
        assert RMSNorm is not None
        assert RotaryPositionalEmbedding is not None

    def test_attention_imports(self):
        from models.layers import (
            BidirectionalAttention,
            CausalSelfAttention,
            FlashCrossAttention,
        )
        assert BidirectionalAttention is not None
        assert CausalSelfAttention is not None
        assert FlashCrossAttention is not None

    @requires_mamba
    def test_mamba_imports(self):
        from models.layers import Mamba2BlockWrapper, BiMambaBlock
        assert Mamba2BlockWrapper is not None
        assert BiMambaBlock is not None

    def test_modeling_imports(self):
        from models.modeling_hybrid import (
            LayerType,
            ModelConfig,
        )
        assert LayerType is not None
        assert ModelConfig is not None

    @requires_mamba
    def test_full_model_imports(self):
        from models.modeling_hybrid import (
            HybridBiMambaEncoder,
            HybridMambaDecoder,
            HybridMambaEncoderDecoder,
        )
        assert HybridBiMambaEncoder is not None
        assert HybridMambaDecoder is not None
        assert HybridMambaEncoderDecoder is not None


class TestRMSNorm:
    """Test RMSNorm implementation."""

    def test_forward_shape(self):
        from models.layers import RMSNorm
        norm = RMSNorm(768).to(DEVICE)
        x = torch.randn(2, 128, 768, device=DEVICE)
        out = norm(x)
        assert out.shape == x.shape

    def test_normalization(self):
        from models.layers import RMSNorm
        norm = RMSNorm(768).to(DEVICE)
        x = torch.randn(2, 128, 768, device=DEVICE)
        out = norm(x)
        # RMS should be approximately 1
        rms = torch.sqrt((out ** 2).mean(dim=-1))
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)


class TestRoPE:
    """Test Rotary Positional Embedding."""

    def test_forward_shape(self):
        from models.layers import RotaryPositionalEmbedding
        rope = RotaryPositionalEmbedding(dim=64, max_seq_len=1024, device=DEVICE)
        q = torch.randn(2, 12, 128, 64, device=DEVICE)
        k = torch.randn(2, 12, 128, 64, device=DEVICE)
        q_rot, k_rot = rope(q, k)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_with_offset(self):
        from models.layers import RotaryPositionalEmbedding
        rope = RotaryPositionalEmbedding(dim=64, max_seq_len=1024, device=DEVICE)
        q = torch.randn(2, 12, 1, 64, device=DEVICE)
        k = torch.randn(2, 12, 1, 64, device=DEVICE)
        q_rot, k_rot = rope(q, k, offset=100)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestMamba2Wrapper:
    """Test Mamba2 block wrapper."""

    @requires_mamba
    def test_forward_shape(self):
        from models.layers import Mamba2BlockWrapper
        block = Mamba2BlockWrapper(
            d_model=256,
            d_state=64,
            device=DEVICE,
            dtype=DTYPE,
        )
        x = torch.randn(2, 128, 256, device=DEVICE, dtype=DTYPE)
        out = block(x)
        assert out.shape == x.shape

    @requires_mamba
    def test_residual_connection(self):
        from models.layers import Mamba2BlockWrapper
        block = Mamba2BlockWrapper(
            d_model=256,
            d_state=64,
            device=DEVICE,
            dtype=DTYPE,
        )
        x = torch.zeros(2, 128, 256, device=DEVICE, dtype=DTYPE)
        out = block(x)
        # With zero input, output should also be close to zero (residual)
        assert out.abs().mean() < 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestBiMamba:
    """Test BiMamba block for encoder."""

    @requires_mamba
    def test_forward_shape(self):
        from models.layers import BiMambaBlock
        block = BiMambaBlock(
            d_model=256,
            d_state=64,
            device=DEVICE,
            dtype=DTYPE,
        )
        x = torch.randn(2, 128, 256, device=DEVICE, dtype=DTYPE)
        out = block(x)
        assert out.shape == x.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestAttentionModules:
    """Test attention modules."""

    def test_causal_self_attention_shape(self):
        from models.layers import CausalSelfAttention
        attn = CausalSelfAttention(
            d_model=256,
            n_heads=4,
            device=DEVICE,
            dtype=DTYPE,
        )
        x = torch.randn(2, 128, 256, device=DEVICE, dtype=DTYPE)
        out, kv = attn(x)
        assert out.shape == x.shape

    def test_bidirectional_attention_shape(self):
        from models.layers import BidirectionalAttention
        attn = BidirectionalAttention(
            d_model=256,
            n_heads=4,
            device=DEVICE,
            dtype=DTYPE,
        )
        x = torch.randn(2, 128, 256, device=DEVICE, dtype=DTYPE)
        out = attn(x)
        assert out.shape == x.shape

    def test_cross_attention_shape(self):
        from models.layers import FlashCrossAttention
        attn = FlashCrossAttention(
            d_model=256,
            n_heads=4,
            device=DEVICE,
            dtype=DTYPE,
        )
        x = torch.randn(2, 64, 256, device=DEVICE, dtype=DTYPE)
        encoder_out = torch.randn(2, 128, 256, device=DEVICE, dtype=DTYPE)
        out = attn(x, encoder_out)
        assert out.shape == x.shape


class TestLayerType:
    """Test LayerType enum and utilities."""

    def test_layer_type_enum(self):
        from models.modeling_hybrid import LayerType, count_layer_types

        types = [LayerType.MAMBA, LayerType.MAMBA, LayerType.HYBRID, LayerType.ATTENTION]
        counts = count_layer_types(types)
        assert counts["mamba"] == 2
        assert counts["hybrid"] == 1
        assert counts["attention"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestHybridEncoder:
    """Test hybrid BiMamba encoder."""

    @requires_mamba
    def test_forward_shape(self):
        from models.modeling_hybrid import HybridBiMambaEncoder
        encoder = HybridBiMambaEncoder(
            vocab_size=1000,
            d_model=256,
            n_layers=4,
            d_state=64,
            n_heads=4,
            attention_ratio=0.25,
            device=DEVICE,
            dtype=DTYPE,
        )
        src_ids = torch.randint(0, 1000, (2, 64), device=DEVICE)
        out = encoder(src_ids)
        assert out.shape == (2, 64, 256)

    @requires_mamba
    def test_layer_counts(self):
        from models.modeling_hybrid import HybridBiMambaEncoder
        encoder = HybridBiMambaEncoder(
            vocab_size=1000,
            d_model=256,
            n_layers=8,
            d_state=64,
            n_heads=4,
            attention_ratio=0.25,
            device=DEVICE,
            dtype=DTYPE,
        )
        counts = encoder.get_layer_counts()
        assert counts["bimamba"] > 0
        assert counts["attention"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestHybridDecoder:
    """Test hybrid Mamba decoder."""

    @requires_mamba
    def test_forward_shape(self):
        from models.modeling_hybrid import HybridMambaDecoder
        decoder = HybridMambaDecoder(
            vocab_size=1000,
            d_model=256,
            n_layers=8,
            d_state=64,
            n_heads=4,
            hybrid_interval=4,
            device=DEVICE,
            dtype=DTYPE,
        )
        tgt_ids = torch.randint(0, 1000, (2, 32), device=DEVICE)
        encoder_out = torch.randn(2, 64, 256, device=DEVICE, dtype=DTYPE)
        logits = decoder(tgt_ids, encoder_out)
        assert logits.shape == (2, 32, 1000)

    @requires_mamba
    def test_hybrid_positions(self):
        from models.modeling_hybrid import HybridMambaDecoder
        decoder = HybridMambaDecoder(
            vocab_size=1000,
            d_model=256,
            n_layers=24,
            d_state=64,
            n_heads=4,
            hybrid_interval=8,
            device=DEVICE,
            dtype=DTYPE,
        )
        # Should have HYBRID at positions 0, 8, 16
        assert 0 in decoder.hybrid_positions
        assert 8 in decoder.hybrid_positions
        assert 16 in decoder.hybrid_positions


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestFullModel:
    """Test full encoder-decoder model."""

    @requires_mamba
    def test_model_creation(self):
        from models import ModelConfig, HybridMambaEncoderDecoder

        config = ModelConfig(
            vocab_size=1000,
            d_model=256,
            encoder_layers=4,
            decoder_layers=8,
            d_state=64,
            n_heads=4,
        )
        model = HybridMambaEncoderDecoder(config=config, device=DEVICE, dtype=DTYPE)
        assert model is not None

    @requires_mamba
    def test_forward_pass(self):
        from models import ModelConfig, HybridMambaEncoderDecoder

        config = ModelConfig(
            vocab_size=1000,
            d_model=256,
            encoder_layers=4,
            decoder_layers=8,
            d_state=64,
            n_heads=4,
        )
        model = HybridMambaEncoderDecoder(config=config, device=DEVICE, dtype=DTYPE)

        src_ids = torch.randint(0, 1000, (2, 64), device=DEVICE)
        tgt_ids = torch.randint(0, 1000, (2, 32), device=DEVICE)

        logits = model(src_ids, tgt_ids)
        assert logits.shape == (2, 32, 1000)

    @requires_mamba
    def test_encode(self):
        from models import ModelConfig, HybridMambaEncoderDecoder

        config = ModelConfig(
            vocab_size=1000,
            d_model=256,
            encoder_layers=4,
            decoder_layers=8,
            d_state=64,
            n_heads=4,
        )
        model = HybridMambaEncoderDecoder(config=config, device=DEVICE, dtype=DTYPE)

        src_ids = torch.randint(0, 1000, (2, 64), device=DEVICE)
        encoder_out = model.encode(src_ids)
        assert encoder_out.shape == (2, 64, 256)

    @requires_mamba
    def test_generation_cache(self):
        from models import ModelConfig, HybridMambaEncoderDecoder

        config = ModelConfig(
            vocab_size=1000,
            d_model=256,
            encoder_layers=4,
            decoder_layers=8,
            d_state=64,
            n_heads=4,
        )
        model = HybridMambaEncoderDecoder(config=config, device=DEVICE, dtype=DTYPE)

        src_ids = torch.randint(0, 1000, (2, 64), device=DEVICE)
        encoder_out = model.encode(src_ids)
        cache = model.init_generation_cache(encoder_out, device=DEVICE, dtype=DTYPE)

        assert "ssm_states" in cache
        assert "kv_caches" in cache
        assert "encoder_output" in cache
        assert "seqlen_offset" in cache

    @requires_mamba
    def test_generate_step(self):
        from models import ModelConfig, HybridMambaEncoderDecoder

        config = ModelConfig(
            vocab_size=1000,
            d_model=256,
            encoder_layers=4,
            decoder_layers=8,
            d_state=64,
            n_heads=4,
        )
        model = HybridMambaEncoderDecoder(config=config, device=DEVICE, dtype=DTYPE)
        model.eval()

        src_ids = torch.randint(0, 1000, (2, 64), device=DEVICE)
        encoder_out = model.encode(src_ids)
        cache = model.init_generation_cache(encoder_out, device=DEVICE, dtype=DTYPE)

        # Generate one token
        input_ids = torch.tensor([[1], [1]], device=DEVICE)  # BOS token
        with torch.no_grad():
            logits, cache = model.generate_step(input_ids, cache)

        assert logits.shape == (2, 1, 1000)
        assert cache["seqlen_offset"] == 1

    @requires_mamba
    def test_generate(self):
        from models import ModelConfig, HybridMambaEncoderDecoder

        config = ModelConfig(
            vocab_size=1000,
            d_model=256,
            encoder_layers=4,
            decoder_layers=8,
            d_state=64,
            n_heads=4,
        )
        model = HybridMambaEncoderDecoder(config=config, device=DEVICE, dtype=DTYPE)
        model.eval()

        src_ids = torch.randint(0, 1000, (2, 64), device=DEVICE)

        with torch.no_grad():
            generated = model.generate(src_ids, max_length=10)

        assert generated.shape[0] == 2
        assert generated.shape[1] <= 11  # BOS + max_length

    @requires_mamba
    def test_parameter_count(self):
        from models import ModelConfig, HybridMambaEncoderDecoder

        config = ModelConfig(
            vocab_size=32000,
            d_model=768,
            encoder_layers=16,
            decoder_layers=24,
            d_state=128,
            n_heads=12,
        )
        model = HybridMambaEncoderDecoder(config=config)

        params = model.num_parameters()
        # Should be around 200M for medium config
        print(f"Parameter count: {params / 1e6:.1f}M")
        assert params > 100_000_000  # At least 100M


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
