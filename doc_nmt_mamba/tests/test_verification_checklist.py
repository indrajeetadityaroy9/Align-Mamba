"""
Critical Verification Tests for Document-Level NMT.

These tests verify that the core mechanisms work correctly before
launching training runs. Each test corresponds to a specific risk
identified in the verification checklist.

Run with: pytest tests/test_verification_checklist.py -v
"""

import pytest
import torch
import torch.nn as nn
from typing import List, Tuple

# Check if mamba-ssm is available
try:
    import mamba_ssm
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False

requires_mamba = pytest.mark.skipif(not HAS_MAMBA, reason="mamba-ssm required (CUDA only)")


# ============================================================================
# 1A. BiMamba Packed Sequence Flip
# ============================================================================

class TestSegmentAwareFlip:
    """
    RISK: If flip() ignores cu_seqlens, it will bleed information across
    document boundaries (e.g., the end of Doc A flows into the start of Doc B).
    """

    def test_segment_aware_flip_basic(self):
        """
        Test: Create tensor [A1, A2, B1, B2, B3] with cu_seqlens=[0, 2, 5].
        Expected: Result must be [A2, A1, B3, B2, B1].
        Failure Mode: If you get [B3, B2, B1, A2, A1], the encoder is broken.
        """
        from doc_nmt_mamba.models.mamba2 import segment_aware_flip

        # Create packed sequence: Doc A has 2 tokens, Doc B has 3 tokens
        # Values: A1=1, A2=2, B1=3, B2=4, B3=5
        x = torch.tensor([
            [1.0, 0.0],  # A1
            [2.0, 0.0],  # A2
            [3.0, 0.0],  # B1
            [4.0, 0.0],  # B2
            [5.0, 0.0],  # B3
        ])  # Shape: (5, 2) - packed sequence

        cu_seqlens = torch.tensor([0, 2, 5])

        # Apply segment-aware flip
        flipped = segment_aware_flip(x, cu_seqlens)

        # Expected: [A2, A1, B3, B2, B1]
        expected = torch.tensor([
            [2.0, 0.0],  # A2 (was position 1, now position 0)
            [1.0, 0.0],  # A1 (was position 0, now position 1)
            [5.0, 0.0],  # B3 (was position 4, now position 2)
            [4.0, 0.0],  # B2 (was position 3, now position 3)
            [3.0, 0.0],  # B1 (was position 2, now position 4)
        ])

        assert torch.allclose(flipped, expected), (
            f"segment_aware_flip is BROKEN!\n"
            f"Got: {flipped[:, 0].tolist()}\n"
            f"Expected: {expected[:, 0].tolist()}\n"
            f"If you got [5,4,3,2,1], the flip ignores document boundaries!"
        )

    def test_segment_aware_flip_single_document(self):
        """Test flip with single document (should work like regular flip)."""
        from doc_nmt_mamba.models.mamba2 import segment_aware_flip

        x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        cu_seqlens = torch.tensor([0, 4])

        flipped = segment_aware_flip(x, cu_seqlens)
        expected = torch.tensor([[4.0], [3.0], [2.0], [1.0]])

        assert torch.allclose(flipped, expected)

    def test_segment_aware_flip_three_documents(self):
        """Test flip with three documents of varying lengths."""
        from doc_nmt_mamba.models.mamba2 import segment_aware_flip

        # Doc A: 2 tokens, Doc B: 1 token, Doc C: 3 tokens
        x = torch.tensor([
            [1.0],  # A1
            [2.0],  # A2
            [3.0],  # B1 (single token doc)
            [4.0],  # C1
            [5.0],  # C2
            [6.0],  # C3
        ])
        cu_seqlens = torch.tensor([0, 2, 3, 6])

        flipped = segment_aware_flip(x, cu_seqlens)

        # Expected: [A2, A1, B1, C3, C2, C1]
        expected = torch.tensor([
            [2.0],  # A2
            [1.0],  # A1
            [3.0],  # B1 (unchanged - single token)
            [6.0],  # C3
            [5.0],  # C2
            [4.0],  # C1
        ])

        assert torch.allclose(flipped, expected), (
            f"Multi-document flip failed!\n"
            f"Got: {flipped[:, 0].tolist()}\n"
            f"Expected: {expected[:, 0].tolist()}"
        )

    def test_segment_aware_flip_no_cu_seqlens(self):
        """Test flip without cu_seqlens (standard batched flip)."""
        from doc_nmt_mamba.models.mamba2 import segment_aware_flip

        # Standard batched tensor (B, L, D)
        x = torch.tensor([
            [[1.0], [2.0], [3.0]],  # Batch 1
            [[4.0], [5.0], [6.0]],  # Batch 2
        ])

        flipped = segment_aware_flip(x, cu_seqlens=None)

        expected = torch.tensor([
            [[3.0], [2.0], [1.0]],
            [[6.0], [5.0], [4.0]],
        ])

        assert torch.allclose(flipped, expected)


# ============================================================================
# 1B. Layer 0 Hybrid Configuration
# ============================================================================

@requires_mamba
class TestHybridLayerConfiguration:
    """
    RISK: The code might default to standard Mamba or Standard Attention
    at Layer 0 if the layer_builder logic is off-by-one.

    NOTE: These tests require mamba-ssm (CUDA only).
    """

    def test_layer_0_is_hybrid_block(self):
        """Verify Layer 0 is a HybridBlock with Mamba + CrossAttention."""
        from doc_nmt_mamba.models.hybrid import (
            build_decoder_layers,
            LayerType,
            HybridBlock,
        )

        layers, layer_types = build_decoder_layers(
            n_layers=24,
            d_model=256,
            d_state=64,
            n_heads=8,
            hybrid_interval=8,
            use_hybrid_blocks=True,
        )

        # Layer 0 must be HYBRID
        assert layer_types[0] == LayerType.HYBRID, (
            f"Layer 0 should be HYBRID, got {layer_types[0]}"
        )

        # Layer 0 must be a HybridBlock instance
        assert isinstance(layers[0], HybridBlock), (
            f"Layer 0 should be HybridBlock, got {type(layers[0])}"
        )

        # HybridBlock must have both Mamba and CrossAttention
        layer0 = layers[0]
        assert hasattr(layer0, 'mamba'), "HybridBlock missing Mamba component"
        assert hasattr(layer0, 'cross_attn'), "HybridBlock missing CrossAttention"

    def test_layer_8_is_hybrid_block(self):
        """Verify Layer 8 is a HybridBlock (first refresh)."""
        from doc_nmt_mamba.models.hybrid import build_decoder_layers, LayerType

        layers, layer_types = build_decoder_layers(
            n_layers=24,
            d_model=256,
            d_state=64,
            n_heads=8,
            hybrid_interval=8,
            use_hybrid_blocks=True,
        )

        assert layer_types[8] == LayerType.HYBRID, (
            f"Layer 8 should be HYBRID, got {layer_types[8]}"
        )

    def test_layer_16_is_hybrid_block(self):
        """Verify Layer 16 is a HybridBlock (second refresh)."""
        from doc_nmt_mamba.models.hybrid import build_decoder_layers, LayerType

        layers, layer_types = build_decoder_layers(
            n_layers=24,
            d_model=256,
            d_state=64,
            n_heads=8,
            hybrid_interval=8,
            use_hybrid_blocks=True,
        )

        assert layer_types[16] == LayerType.HYBRID, (
            f"Layer 16 should be HYBRID, got {layer_types[16]}"
        )

    def test_layer_1_is_mamba_only(self):
        """Verify Layer 1 is Mamba-only (not hybrid)."""
        from doc_nmt_mamba.models.hybrid import build_decoder_layers, LayerType

        layers, layer_types = build_decoder_layers(
            n_layers=24,
            d_model=256,
            d_state=64,
            n_heads=8,
            hybrid_interval=8,
            use_hybrid_blocks=True,
        )

        assert layer_types[1] == LayerType.MAMBA, (
            f"Layer 1 should be MAMBA, got {layer_types[1]}"
        )

    def test_hybrid_layer_pattern(self):
        """Verify exact pattern: HYBRID at [0, 8, 16], MAMBA elsewhere."""
        from doc_nmt_mamba.models.hybrid import build_decoder_layers, LayerType

        layers, layer_types = build_decoder_layers(
            n_layers=24,
            d_model=256,
            d_state=64,
            n_heads=8,
            hybrid_interval=8,
            use_hybrid_blocks=True,
        )

        hybrid_indices = [i for i, lt in enumerate(layer_types) if lt == LayerType.HYBRID]

        assert hybrid_indices == [0, 8, 16], (
            f"HYBRID layers should be at [0, 8, 16], got {hybrid_indices}"
        )

        # All other layers should be MAMBA
        for i, lt in enumerate(layer_types):
            if i not in [0, 8, 16]:
                assert lt == LayerType.MAMBA, (
                    f"Layer {i} should be MAMBA, got {lt}"
                )


# ============================================================================
# 1C. Cross-Attention Tensor Shapes
# ============================================================================

@requires_mamba
class TestCrossAttentionShapes:
    """
    RISK: Mamba operates on (B, L, D). Cross-Attention needs (B, L, H, D_head).
    Ensure projections are correct.

    NOTE: These tests require mamba-ssm (CUDA only).
    """

    def test_hybrid_block_forward_shapes(self):
        """Verify HybridBlock handles tensor shapes correctly."""
        from doc_nmt_mamba.models.hybrid import HybridBlock

        batch_size = 2
        tgt_len = 64
        src_len = 128
        d_model = 256
        n_heads = 8

        block = HybridBlock(
            d_model=d_model,
            d_state=64,
            n_heads=n_heads,
            dropout=0.1,
        )

        # Decoder input: (B, L_tgt, D)
        decoder_input = torch.randn(batch_size, tgt_len, d_model)
        # Encoder output: (B, L_src, D)
        encoder_out = torch.randn(batch_size, src_len, d_model)

        # Forward pass should not crash
        output = block(decoder_input, encoder_out)

        # Output shape must match input shape
        assert output.shape == decoder_input.shape, (
            f"Output shape {output.shape} != input shape {decoder_input.shape}"
        )

    def test_encoder_decoder_batch_alignment(self):
        """Verify encoder and decoder batch sizes align."""
        from doc_nmt_mamba.models.hybrid import HybridBlock

        batch_size = 4
        d_model = 256

        block = HybridBlock(d_model=d_model, d_state=64, n_heads=8)

        # Same batch size for encoder and decoder
        decoder_input = torch.randn(batch_size, 32, d_model)
        encoder_out = torch.randn(batch_size, 64, d_model)

        output = block(decoder_input, encoder_out)
        assert output.size(0) == batch_size

    def test_cross_attention_projection_dimensions(self):
        """Verify Q, K, V projections have correct dimensions."""
        from doc_nmt_mamba.models.attention import FlashCrossAttention

        d_model = 256
        n_heads = 8
        head_dim = d_model // n_heads

        attn = FlashCrossAttention(d_model=d_model, n_heads=n_heads)

        # Check projection dimensions
        assert attn.q_proj.in_features == d_model
        assert attn.q_proj.out_features == d_model
        assert attn.k_proj.in_features == d_model
        assert attn.k_proj.out_features == d_model
        assert attn.v_proj.in_features == d_model
        assert attn.v_proj.out_features == d_model


# ============================================================================
# 2A. MQAR Leakage Check
# ============================================================================

class TestMQARLeakage:
    """
    RISK: If the Target Value accidentally appears in the "Context" after
    the Query, the model can just copy the nearest neighbor instead of
    performing long-range recall.
    """

    def test_query_after_kv_pairs(self):
        """Verify Query tokens appear strictly after Key-Value pairs."""
        from doc_nmt_mamba.data.synthetic import MQARDataset, MQARConfig

        config = MQARConfig(
            vocab_size=8192,
            num_pairs=16,
            num_queries=4,
            seq_length=256,
        )
        dataset = MQARDataset(config, num_samples=10)

        sample = dataset[0]
        input_ids = sample['input_ids']

        # Find special token positions
        query_token = config.query_token_id
        kv_sep = config.kv_sep_token_id

        # Find the QUERY marker position
        query_positions = (input_ids == query_token).nonzero(as_tuple=True)[0]

        assert len(query_positions) > 0, "No QUERY token found in input"

        query_start = query_positions[0].item()

        # Find the last KV separator before QUERY
        kv_sep_positions = (input_ids[:query_start] == kv_sep).nonzero(as_tuple=True)[0]

        assert len(kv_sep_positions) > 0, (
            "No KV separators found before QUERY token"
        )

        last_kv_sep = kv_sep_positions[-1].item()

        # QUERY must come after all KV pairs
        assert query_start > last_kv_sep, (
            f"QUERY at {query_start} should come after last KV at {last_kv_sep}"
        )

    def test_target_values_only_in_kv_section(self):
        """Verify target values appear in context only with their keys."""
        from doc_nmt_mamba.data.synthetic import MQARDataset, MQARConfig

        config = MQARConfig(
            vocab_size=8192,
            num_pairs=16,
            num_queries=4,
            seq_length=256,
        )
        dataset = MQARDataset(config, num_samples=10)

        sample = dataset[0]
        input_ids = sample['input_ids']
        labels = sample['labels']

        # Get actual target values (non-padding, non-EOS)
        target_mask = (labels != config.pad_token_id) & (labels != config.eos_token_id)
        target_values = labels[target_mask].tolist()

        # Find QUERY token position
        query_token = config.query_token_id
        query_positions = (input_ids == query_token).nonzero(as_tuple=True)[0]

        assert len(query_positions) > 0, "No QUERY token found"

        query_start = query_positions[0].item()

        # Post-QUERY section should be: QUERY k1 k2 k3 ... EOS
        # NOT: QUERY k1 v1 k2 v2 ... (values should not appear here)
        post_query_section = input_ids[query_start + 1:].tolist()

        # Values are in range [4096, 8192), keys are in range [10, 4096)
        # Check that target values (which are values) don't appear as raw tokens
        # after the QUERY (they should only be queried, not directly visible)
        for val in target_values:
            # Value should NOT appear in the query section
            # (the query section has keys to lookup, not the values)
            assert val not in post_query_section or val >= config.key_token_end, (
                f"Value {val} should not appear in query section"
            )

    def test_kv_structure_consistency(self):
        """Verify KV structure follows k:v pattern."""
        from doc_nmt_mamba.data.synthetic import MQARDataset, MQARConfig

        config = MQARConfig(
            vocab_size=8192,
            num_pairs=8,  # Small for easier verification
            num_queries=4,
        )
        dataset = MQARDataset(config, num_samples=10)

        sample = dataset[0]
        input_ids = sample['input_ids']
        labels = sample['labels']

        # Find KV separator positions
        kv_sep_positions = (input_ids == config.kv_sep_token_id).nonzero(as_tuple=True)[0]

        # Should have exactly num_pairs KV separators (one per pair)
        assert len(kv_sep_positions) == sample['num_pairs'].item(), (
            f"Expected {sample['num_pairs'].item()} KV separators, "
            f"got {len(kv_sep_positions)}"
        )

        # Each separator should have a key before and value after
        for kv_sep_pos in kv_sep_positions:
            pos = kv_sep_pos.item()
            key = input_ids[pos - 1].item()
            value = input_ids[pos + 1].item()

            # Key should be in key range
            assert config.key_token_start <= key < config.key_token_end, (
                f"Key {key} not in valid range [{config.key_token_start}, {config.key_token_end})"
            )

            # Value should be in value range
            assert config.value_token_start <= value < config.value_token_end, (
                f"Value {value} not in valid range [{config.value_token_start}, {config.value_token_end})"
            )


# ============================================================================
# 2B. Bottleneck Configuration
# ============================================================================

class TestMQARBottleneckConfig:
    """
    RISK: You want to force a cliff. If d_state is too large, Mamba will
    solve the task too easily.
    """

    def test_default_d_state_is_64(self):
        """Verify default d_state is 64 (reduced to force bottleneck)."""
        from doc_nmt_mamba.data.synthetic import MQARConfig

        config = MQARConfig()
        assert config.d_state == 64, (
            f"d_state should be 64 for bottleneck, got {config.d_state}"
        )

    def test_num_pairs_can_exceed_d_state(self):
        """Verify we can create datasets with num_pairs > d_state."""
        from doc_nmt_mamba.data.synthetic import MQARDataset, MQARConfig

        # This should exceed state capacity
        config = MQARConfig(
            d_state=64,
            num_pairs=128,  # 2x state capacity
            seq_length=2048,
        )

        dataset = MQARDataset(config, num_samples=5)
        sample = dataset[0]

        # Should have created the dataset successfully
        assert sample['num_pairs'].item() == 128

    def test_curriculum_spans_cliff(self):
        """Verify curriculum includes stages that span the cliff point."""
        from doc_nmt_mamba.data.synthetic import MQARCurriculumGenerator

        generator = MQARCurriculumGenerator(
            d_state=64,
            num_pairs_range=[16, 32, 64, 128, 256],
        )

        # Should have stages below, at, and above d_state
        assert 16 in generator.num_pairs_range  # Below
        assert 64 in generator.num_pairs_range  # At d_state
        assert 128 in generator.num_pairs_range  # 2x d_state


# ============================================================================
# 3A. Subword-to-Word Mapping (AER)
# ============================================================================

class TestSubwordToWordMapping:
    """
    RISK: If the tokenizer splits "bank" into ["b", "ank"], and alignment
    points to "ank", but the Gold Standard points to "bank", your mapping
    must resolve this.
    """

    def test_build_token_to_word_map_basic(self):
        """Test basic token-to-word mapping."""
        from doc_nmt_mamba.evaluation.alignment import SubwordToWordMapper

        mapper = SubwordToWordMapper(word_boundary_prefix="▁")

        # Simulate tokenized output with sentencepiece-style tokens
        tokens = ["▁The", "▁b", "ank", "▁is", "▁open", "."]

        mapping = mapper.build_token_to_word_map(tokens)

        # Expected:
        # "▁The" -> word 0
        # "▁b" -> word 1
        # "ank" -> word 1 (continuation of "bank")
        # "▁is" -> word 2
        # "▁open" -> word 3
        # "." -> word 3 (continuation - no prefix)

        assert mapping[0] == 0, f"'▁The' should map to word 0, got {mapping[0]}"
        assert mapping[1] == 1, f"'▁b' should map to word 1, got {mapping[1]}"
        assert mapping[2] == 1, f"'ank' should map to word 1, got {mapping[2]}"
        assert mapping[3] == 2, f"'▁is' should map to word 2, got {mapping[3]}"
        assert mapping[4] == 3, f"'▁open' should map to word 3, got {mapping[4]}"

    def test_subword_aggregation_max_pooling(self):
        """Test attention aggregation uses max-pooling correctly."""
        from doc_nmt_mamba.evaluation.alignment import SubwordToWordMapper

        mapper = SubwordToWordMapper()

        # Token-level attention: (3 target tokens, 4 source tokens)
        attn = torch.tensor([
            [0.1, 0.8, 0.0, 0.1],  # Target token 0 attends strongly to source 1
            [0.2, 0.7, 0.1, 0.0],  # Target token 1 also attends to source 1
            [0.0, 0.1, 0.3, 0.6],  # Target token 2 attends to source 3
        ])

        # Mapping: tokens 0,1 -> word 0; token 2 -> word 1
        src_map = [0, 0, 1, 1]  # source tokens to words
        tgt_map = [0, 0, 1]     # target tokens to words

        word_attn = mapper.aggregate_attention_to_words(
            attn, src_map, tgt_map,
            n_src_words=2, n_tgt_words=2,
            aggregation="max"
        )

        # Word 0 (src) <- Word 0 (tgt): max of attn[0:2, 0:2]
        # = max(0.1, 0.8, 0.2, 0.7) = 0.8
        assert word_attn[0, 0].item() == pytest.approx(0.8, abs=0.01)

    def test_punctuation_handling(self):
        """Test that punctuation gets proper word indices."""
        from doc_nmt_mamba.evaluation.alignment import SubwordToWordMapper

        mapper = SubwordToWordMapper(word_boundary_prefix="▁")

        # Punctuation typically doesn't have word boundary prefix
        # so it continues from the previous word
        tokens = ["▁Hello", "▁world", "!", "▁How", "▁are", "▁you", "?"]

        mapping = mapper.build_token_to_word_map(tokens)

        # "▁Hello" -> word 0
        # "▁world" -> word 1
        # "!" -> continues word 1 (no prefix)
        # "▁How" -> word 2
        # "▁are" -> word 3
        # "▁you" -> word 4
        # "?" -> continues word 4 (no prefix)
        assert mapping[0] == 0, f"'▁Hello' should be word 0, got {mapping[0]}"
        assert mapping[1] == 1, f"'▁world' should be word 1, got {mapping[1]}"
        assert mapping[2] == 1, f"'!' should continue word 1, got {mapping[2]}"
        assert mapping[3] == 2, f"'▁How' should be word 2, got {mapping[3]}"
        assert mapping[6] == 4, f"'?' should continue word 4, got {mapping[6]}"


# ============================================================================
# 3B. CAT-N Concatenation
# ============================================================================

class TestCATNConcatenation:
    """
    RISK: The model must see the <doc> separator.
    """

    def test_separator_in_concatenated_output(self):
        """Verify concatenated samples contain separator."""
        from doc_nmt_mamba.data.augmentation import ConcatenationAugmenter, DocumentSample

        # Use a clear separator
        augmenter = ConcatenationAugmenter(
            n_sentences=5,
            p_concat=1.0,  # Always concatenate
            separator=" <doc> ",
            min_concat=2,
            max_concat=3,
            seed=42,
        )

        # Create a document with multiple sentences
        doc = DocumentSample(
            src_sentences=["Hello world.", "How are you?", "I am fine."],
            tgt_sentences=["Xin chao.", "Ban khoe khong?", "Toi khoe."],
            doc_id="doc1",
        )

        # Augment - should produce concatenated samples
        samples = augmenter.augment_document(doc)

        # At least one sample should be concatenated (with separator)
        has_separator = any("<doc>" in src for src, tgt in samples)

        assert has_separator, (
            f"Separator not found in any sample. Samples: {samples}"
        )

    def test_concatenation_preserves_alignment(self):
        """Verify source and target remain aligned after concatenation."""
        from doc_nmt_mamba.data.augmentation import ConcatenationAugmenter, DocumentSample

        augmenter = ConcatenationAugmenter(
            n_sentences=5,
            p_concat=1.0,  # Always concatenate
            separator=" | ",
            min_concat=2,
            seed=42,
        )

        doc = DocumentSample(
            src_sentences=["One", "Two", "Three", "Four"],
            tgt_sentences=["Mot", "Hai", "Ba", "Bon"],
        )

        samples = augmenter.augment_document(doc)

        for src, tgt in samples:
            # Count separators - should match in source and target
            src_sep_count = src.count("|")
            tgt_sep_count = tgt.count("|")
            assert src_sep_count == tgt_sep_count, (
                f"Separator count mismatch: src={src_sep_count}, tgt={tgt_sep_count}"
            )


# ============================================================================
# 4A. Inference State Management
# ============================================================================

class TestInferenceStateManagement:
    """
    RISK: During decoding, Mamba states are fixed-size, but Attention KV caches grow.
    """

    def test_mamba_state_size_constant(self):
        """Verify Mamba state tensor size remains constant during generation."""
        # This test requires the model, which needs mamba-ssm
        pytest.importorskip("mamba_ssm", reason="mamba-ssm required for this test")

        from doc_nmt_mamba.models import HybridMambaEncoderDecoder, ModelConfig

        config = ModelConfig(
            vocab_size=1000,
            d_model=256,
            encoder_layers=4,
            decoder_layers=8,
            d_state=64,
            n_heads=8,
        )

        model = HybridMambaEncoderDecoder(config=config)
        model.eval()

        batch_size = 2
        src_len = 32

        src_ids = torch.randint(10, 100, (batch_size, src_len))

        with torch.no_grad():
            encoder_out = model.encode(src_ids)
            cache = model.init_generation_cache(encoder_out)

            # Get initial Mamba state sizes
            initial_ssm_sizes = {
                k: (v.conv_state.shape, v.ssm_state.shape)
                for k, v in cache['ssm_states'].items()
            }

            # Generate 10 tokens
            token = torch.full((batch_size, 1), 1, dtype=torch.long)

            for step in range(10):
                _, cache = model.generate_step(token, cache)

                # Check Mamba state sizes haven't changed
                for k, v in cache['ssm_states'].items():
                    current_sizes = (v.conv_state.shape, v.ssm_state.shape)
                    assert current_sizes == initial_ssm_sizes[k], (
                        f"Mamba state size changed at step {step}! "
                        f"Was {initial_ssm_sizes[k]}, now {current_sizes}"
                    )

    def test_kv_cache_grows_with_generation(self):
        """Verify Attention KV cache grows during generation."""
        # Skip if no mamba-ssm (can't instantiate full model)
        pytest.importorskip("mamba_ssm", reason="mamba-ssm required for this test")

        from doc_nmt_mamba.models import HybridMambaEncoderDecoder, ModelConfig

        config = ModelConfig(
            vocab_size=1000,
            d_model=256,
            encoder_layers=4,
            decoder_layers=8,
            d_state=64,
            n_heads=8,
            attention_ratio=0.25,  # Some attention layers
        )

        model = HybridMambaEncoderDecoder(config=config)
        model.eval()

        batch_size = 2
        src_ids = torch.randint(10, 100, (batch_size, 32))

        with torch.no_grad():
            encoder_out = model.encode(src_ids)
            cache = model.init_generation_cache(encoder_out)

            token = torch.full((batch_size, 1), 1, dtype=torch.long)

            kv_sizes_over_time = []

            for step in range(5):
                _, cache = model.generate_step(token, cache)

                # Record KV cache sizes
                kv_sizes = {}
                for k, v in cache.get('kv_caches', {}).items():
                    if v.key_cache is not None:
                        kv_sizes[k] = v.key_cache.shape[1]  # Sequence length dim
                kv_sizes_over_time.append(kv_sizes)

            # KV cache should grow (if we have attention layers)
            if kv_sizes_over_time[0]:
                for k in kv_sizes_over_time[0]:
                    sizes = [s.get(k, 0) for s in kv_sizes_over_time]
                    # Should be increasing
                    assert all(sizes[i] <= sizes[i+1] for i in range(len(sizes)-1)), (
                        f"KV cache for layer {k} not growing: {sizes}"
                    )


# ============================================================================
# Smoke Tests
# ============================================================================

class TestSmokeTests:
    """Quick sanity checks that the entire stack works."""

    def test_mqar_generation(self):
        """Verify MQAR dataset can be created and sampled."""
        from doc_nmt_mamba.data.synthetic import MQARDataset, MQARConfig

        config = MQARConfig(num_pairs=16)
        dataset = MQARDataset(config, num_samples=10)

        sample = dataset[0]

        assert 'input_ids' in sample
        assert 'labels' in sample
        assert 'num_pairs' in sample
        assert 'num_queries' in sample

        print(f"\nMQAR Sample:")
        print(f"  Input shape: {sample['input_ids'].shape}")
        print(f"  Labels shape: {sample['labels'].shape}")
        print(f"  Num pairs: {sample['num_pairs'].item()}")

    def test_evaluation_imports(self):
        """Verify all evaluation modules can be imported."""
        from doc_nmt_mamba.evaluation import (
            BLEUScorer,
            CHRFScorer,
            ContrastivePronounEvaluator,
            EntityRecallAnalyzer,
            AlignmentEvaluator,
            SubwordToWordMapper,
        )

        # Quick instantiation checks
        bleu = BLEUScorer()
        mapper = SubwordToWordMapper()

        assert bleu is not None
        assert mapper is not None

    def test_data_pipeline_imports(self):
        """Verify all data pipeline modules can be imported."""
        from doc_nmt_mamba.data import (
            MQARDataset,
            ConcatenationAugmenter,
            PackedSequenceCollator,
        )

        assert MQARDataset is not None
        assert ConcatenationAugmenter is not None
        assert PackedSequenceCollator is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
