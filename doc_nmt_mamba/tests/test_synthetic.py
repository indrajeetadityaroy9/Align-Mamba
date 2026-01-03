"""
Unit tests for MQAR synthetic data generation.

Tests:
- MQAR dataset generation correctness
- Curriculum generation for num_pairs sweep
- State capacity stress test properties
- Collation and accuracy computation
"""

import pytest
import torch
import random

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.synthetic import (
    MQARConfig,
    MQARDataset,
    MQARCurriculumGenerator,
    MQARCollator,
    compute_mqar_accuracy,
    create_mqar_decoder_only_format,
)


class TestMQARConfig:
    """Test MQAR configuration."""

    def test_default_config(self):
        """Verify default config values for state capacity testing."""
        config = MQARConfig()

        # Critical: d_state=64 to force cliff (from plan)
        assert config.d_state == 64, "d_state should be 64 to force bottleneck"
        assert config.vocab_size == 8192, "vocab_size should be 8192"

        # Special tokens should be in reserved range
        assert config.pad_token_id < 10
        assert config.bos_token_id < 10
        assert config.eos_token_id < 10
        assert config.sep_token_id < 10
        assert config.kv_sep_token_id < 10
        assert config.query_token_id < 10

        # Key/value ranges should not overlap with special tokens
        assert config.key_token_start >= 10
        assert config.value_token_start >= 10

    def test_key_value_ranges_non_overlapping(self):
        """Keys and values should come from separate vocab ranges."""
        config = MQARConfig()

        # Key range: [10, 4096)
        # Value range: [4096, 8192)
        assert config.key_token_end <= config.value_token_start, \
            "Key and value ranges should not overlap"


class TestMQARDataset:
    """Test MQAR dataset generation."""

    def test_dataset_creation(self):
        """Basic dataset creation test."""
        config = MQARConfig(num_pairs=32, seq_length=512, num_queries=4)
        dataset = MQARDataset(config=config, num_samples=100, seed=42)

        assert len(dataset) == 100
        sample = dataset[0]

        assert "input_ids" in sample
        assert "labels" in sample
        assert "num_pairs" in sample
        assert "num_queries" in sample
        assert "query_positions" in sample

    def test_sample_format(self):
        """Verify sample format matches MQAR specification."""
        config = MQARConfig(num_pairs=16, seq_length=256, num_queries=4)
        dataset = MQARDataset(config=config, num_samples=10, seed=42)
        sample = dataset[0]

        input_ids = sample["input_ids"]
        labels = sample["labels"]

        # Input should start with BOS
        assert input_ids[0].item() == config.bos_token_id, "Input should start with BOS"

        # Input should contain QUERY token
        query_pos = (input_ids == config.query_token_id).nonzero(as_tuple=True)[0]
        assert len(query_pos) == 1, "Should have exactly one QUERY token"

        # Labels should be mostly PAD until answer positions
        num_pads = (labels == config.pad_token_id).sum().item()
        assert num_pads > 0, "Labels should have PAD tokens before answers"

    def test_key_value_pairs_unique_keys(self):
        """Keys should be unique within each sample."""
        config = MQARConfig(num_pairs=64, seq_length=1024, num_queries=8)
        dataset = MQARDataset(config=config, num_samples=50, seed=42)

        for i in range(len(dataset)):
            sample = dataset[i]
            input_ids = sample["input_ids"].tolist()

            # Extract keys (tokens before kv_sep_token_id)
            keys = []
            for j in range(len(input_ids) - 1):
                if input_ids[j + 1] == config.kv_sep_token_id:
                    keys.append(input_ids[j])

            # Check uniqueness
            assert len(keys) == len(set(keys)), f"Sample {i} has duplicate keys"

    def test_queries_are_valid_keys(self):
        """Query tokens should be keys from the KV pairs."""
        config = MQARConfig(num_pairs=32, seq_length=512, num_queries=8)
        dataset = MQARDataset(config=config, num_samples=20, seed=42)

        for i in range(len(dataset)):
            sample = dataset[i]
            input_ids = sample["input_ids"].tolist()

            # Find QUERY token position
            query_pos = input_ids.index(config.query_token_id)

            # Extract keys from KV section
            keys = set()
            for j in range(len(input_ids) - 1):
                if input_ids[j + 1] == config.kv_sep_token_id:
                    keys.add(input_ids[j])

            # Extract query keys (tokens after QUERY, before EOS)
            eos_pos = input_ids.index(config.eos_token_id)
            query_keys = input_ids[query_pos + 1:eos_pos]

            # All query keys should be valid keys
            for qk in query_keys:
                assert qk in keys, f"Query key {qk} not found in KV pairs"

    def test_target_values_match_kv_mapping(self):
        """Target values should match the key-value mapping."""
        config = MQARConfig(num_pairs=32, seq_length=512, num_queries=4)
        dataset = MQARDataset(config=config, num_samples=20, seed=42)

        for i in range(len(dataset)):
            sample = dataset[i]
            input_ids = sample["input_ids"].tolist()
            labels = sample["labels"].tolist()

            # Build KV mapping from input
            kv_map = {}
            j = 1  # Skip BOS
            while j < len(input_ids) - 2:
                if input_ids[j + 1] == config.kv_sep_token_id:
                    key = input_ids[j]
                    value = input_ids[j + 2]
                    kv_map[key] = value
                    j += 3
                else:
                    j += 1

            # Find query and answer positions
            query_pos = input_ids.index(config.query_token_id)
            eos_pos = input_ids.index(config.eos_token_id)
            query_keys = input_ids[query_pos + 1:eos_pos]

            # Get target values (non-PAD, non-special tokens in labels)
            target_values = [
                v for v in labels
                if v != config.pad_token_id and v != config.eos_token_id
            ]

            # Verify mapping
            for qk, tv in zip(query_keys, target_values):
                expected = kv_map.get(qk)
                assert tv == expected, f"Expected value {expected} for key {qk}, got {tv}"

    def test_reproducibility(self):
        """Same seed should produce identical datasets."""
        config = MQARConfig(num_pairs=32, seq_length=512)
        dataset1 = MQARDataset(config=config, num_samples=100, seed=42)
        dataset2 = MQARDataset(config=config, num_samples=100, seed=42)

        for i in range(len(dataset1)):
            s1 = dataset1[i]
            s2 = dataset2[i]
            assert torch.equal(s1["input_ids"], s2["input_ids"]), f"Sample {i} differs"
            assert torch.equal(s1["labels"], s2["labels"]), f"Labels {i} differ"

    def test_sequence_length_constraint(self):
        """All sequences should respect max sequence length."""
        config = MQARConfig(num_pairs=32, seq_length=256)
        dataset = MQARDataset(config=config, num_samples=50, seed=42)

        for i in range(len(dataset)):
            sample = dataset[i]
            assert len(sample["input_ids"]) == config.seq_length
            assert len(sample["labels"]) == config.seq_length


class TestMQARCurriculum:
    """Test curriculum generation for state capacity sweep."""

    def test_curriculum_generator_creation(self):
        """Test curriculum generator with default config."""
        generator = MQARCurriculumGenerator(
            d_state=64,
            vocab_size=8192,
            num_pairs_range=[16, 32, 64, 128, 256],
        )

        assert generator.d_state == 64
        assert len(generator.num_pairs_range) == 5

    def test_generate_stage(self):
        """Test generating a single curriculum stage."""
        generator = MQARCurriculumGenerator(d_state=64)
        dataset = generator.generate_stage(
            num_pairs=32,
            seq_length=512,
            num_samples=100,
        )

        assert len(dataset) == 100
        assert dataset.config.num_pairs == 32
        assert dataset.config.seq_length == 512

    def test_full_curriculum_generation(self):
        """Test generating full curriculum across all pair counts."""
        generator = MQARCurriculumGenerator(
            d_state=64,
            num_pairs_range=[16, 32, 64, 128],
            samples_per_stage=100,
        )

        stages = generator.generate_full_curriculum(seq_length=512)

        assert len(stages) == 4
        assert 16 in stages
        assert 32 in stages
        assert 64 in stages
        assert 128 in stages

        # Verify each stage has correct num_pairs
        for num_pairs, dataset in stages.items():
            assert dataset.config.num_pairs == num_pairs

    def test_state_capacity_sweep(self):
        """Test sweep across num_pairs and seq_lengths."""
        generator = MQARCurriculumGenerator(
            d_state=64,
            seq_lengths=[256, 512],
            num_pairs_range=[16, 32, 64],
        )

        sweep = generator.generate_state_capacity_sweep(num_samples=50)

        # Should skip configs where num_pairs * 4 > seq_length
        # For seq_length=256: num_pairs can be 16, 32, 64 (64*4=256, edge case)
        # For seq_length=512: num_pairs can be 16, 32, 64
        assert len(sweep) > 0

        for num_pairs, seq_length, dataset in sweep:
            # Verify constraint: num_pairs * 4 <= seq_length
            assert num_pairs * 4 <= seq_length
            assert dataset.config.num_pairs == num_pairs
            assert dataset.config.seq_length == seq_length

    def test_curriculum_increasing_difficulty(self):
        """Verify curriculum increases difficulty (num_pairs)."""
        generator = MQARCurriculumGenerator(
            d_state=64,
            num_pairs_range=[16, 32, 64, 128, 256],
        )

        # num_pairs_range should be in ascending order for curriculum
        for i in range(len(generator.num_pairs_range) - 1):
            assert generator.num_pairs_range[i] < generator.num_pairs_range[i + 1]


class TestMQARCollator:
    """Test MQAR batch collation."""

    def test_collator_basic(self):
        """Test basic collation."""
        config = MQARConfig(num_pairs=16, seq_length=128)
        dataset = MQARDataset(config=config, num_samples=8, seed=42)
        collator = MQARCollator(pad_token_id=config.pad_token_id)

        batch = [dataset[i] for i in range(4)]
        result = collator(batch)

        assert result["input_ids"].shape == (4, 128)
        assert result["labels"].shape == (4, 128)
        assert result["attention_mask"].shape == (4, 128)
        assert result["label_mask"].shape == (4, 128)

    def test_attention_mask_correctness(self):
        """Attention mask should be 1 for non-pad, 0 for pad."""
        config = MQARConfig(num_pairs=16, seq_length=128)
        dataset = MQARDataset(config=config, num_samples=4, seed=42)
        collator = MQARCollator(pad_token_id=config.pad_token_id)

        batch = [dataset[i] for i in range(4)]
        result = collator(batch)

        # Verify mask aligns with input
        for i in range(4):
            input_ids = result["input_ids"][i]
            mask = result["attention_mask"][i]

            for j in range(len(input_ids)):
                if input_ids[j] == config.pad_token_id:
                    assert mask[j] == 0
                else:
                    assert mask[j] == 1


class TestMQARAccuracy:
    """Test accuracy computation for MQAR."""

    def test_perfect_predictions(self):
        """100% accuracy when predictions match labels exactly."""
        # Create mock data
        labels = torch.tensor([[0, 0, 0, 100, 200, 300, 2]])  # PAD PAD PAD v1 v2 v3 EOS
        predictions = torch.tensor([[0, 0, 0, 100, 200, 300, 2]])
        label_mask = torch.tensor([[0, 0, 0, 1, 1, 1, 0]])  # Only score values

        metrics = compute_mqar_accuracy(predictions, labels, label_mask)

        assert metrics["token_accuracy"] == 1.0
        assert metrics["sample_accuracy"] == 1.0

    def test_zero_accuracy(self):
        """0% accuracy when all predictions wrong."""
        labels = torch.tensor([[0, 0, 0, 100, 200, 300, 2]])
        predictions = torch.tensor([[0, 0, 0, 999, 999, 999, 2]])  # All wrong
        label_mask = torch.tensor([[0, 0, 0, 1, 1, 1, 0]])

        metrics = compute_mqar_accuracy(predictions, labels, label_mask)

        assert metrics["token_accuracy"] == 0.0
        assert metrics["sample_accuracy"] == 0.0

    def test_partial_accuracy(self):
        """Partial accuracy with some correct predictions."""
        labels = torch.tensor([[0, 0, 100, 200, 300, 2]])
        predictions = torch.tensor([[0, 0, 100, 999, 300, 2]])  # 2/3 correct
        label_mask = torch.tensor([[0, 0, 1, 1, 1, 0]])

        metrics = compute_mqar_accuracy(predictions, labels, label_mask)

        assert abs(metrics["token_accuracy"] - 2/3) < 0.01

    def test_accuracy_with_logits(self):
        """Test accuracy computation with logit inputs (B, L, V)."""
        vocab_size = 100
        labels = torch.tensor([[0, 0, 50, 60, 70, 2]])
        label_mask = torch.tensor([[0, 0, 1, 1, 1, 0]])

        # Create logits where argmax matches labels
        logits = torch.zeros(1, 6, vocab_size)
        logits[0, 2, 50] = 10.0  # High logit for correct class
        logits[0, 3, 60] = 10.0
        logits[0, 4, 70] = 10.0

        metrics = compute_mqar_accuracy(logits, labels, label_mask)

        assert metrics["token_accuracy"] == 1.0


class TestMQARDecoderOnlyFormat:
    """Test conversion to decoder-only format for fair baseline."""

    def test_decoder_only_conversion(self):
        """Test conversion maintains sequence structure."""
        config = MQARConfig(num_pairs=16, seq_length=128)
        dataset = MQARDataset(config=config, num_samples=4, seed=42)
        collator = MQARCollator(pad_token_id=config.pad_token_id)

        batch = [dataset[i] for i in range(4)]
        collated = collator(batch)
        decoder_batch = create_mqar_decoder_only_format(collated, config)

        # Should have same input_ids
        assert torch.equal(decoder_batch["input_ids"], collated["input_ids"])

        # Labels should be shifted for next-token prediction
        assert decoder_batch["labels"].shape == collated["labels"].shape

        # Label mask should still focus on answer positions
        assert "label_mask" in decoder_batch


class TestStateCapacityCliff:
    """Test properties that demonstrate state capacity limitation.

    CRITICAL: These tests verify the synthetic data has properties
    that will expose Mamba's TC⁰ limitation when num_pairs > d_state.
    """

    def test_cliff_regime_dataset_creation(self):
        """Create datasets at cliff regime (num_pairs >> d_state)."""
        d_state = 64  # From plan: reduced to force bottleneck

        # Dataset at cliff: num_pairs = 4 * d_state = 256
        config = MQARConfig(
            d_state=d_state,
            num_pairs=256,
            seq_length=2048,
            num_queries=8,
        )
        dataset = MQARDataset(config=config, num_samples=100, seed=42)

        assert len(dataset) == 100
        assert dataset.config.num_pairs == 256
        assert dataset.config.num_pairs > config.d_state  # Beyond state capacity

    def test_curriculum_stages_span_cliff(self):
        """Curriculum should span from below to above d_state."""
        generator = MQARCurriculumGenerator(
            d_state=64,
            num_pairs_range=[16, 32, 64, 128, 256, 512],
        )

        # Should have stages both below and above d_state
        below_d_state = [n for n in generator.num_pairs_range if n < 64]
        above_d_state = [n for n in generator.num_pairs_range if n > 64]

        assert len(below_d_state) >= 2, "Should have multiple stages below d_state"
        assert len(above_d_state) >= 2, "Should have multiple stages above d_state"

    def test_extreme_difficulty_regime(self):
        """Test extreme difficulty: num_pairs = 8 * d_state."""
        config = MQARConfig(
            d_state=64,
            num_pairs=512,  # 8x d_state
            seq_length=4096,
            num_queries=8,
        )
        dataset = MQARDataset(config=config, num_samples=50, seed=42)

        # Verify we can create samples at this extreme
        sample = dataset[0]
        assert sample["num_pairs"].item() <= 512

        # At this regime:
        # - Pure Mamba: Expected accuracy < 10% (from plan)
        # - Hybrid with cross-attention: Expected accuracy > 75%


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
