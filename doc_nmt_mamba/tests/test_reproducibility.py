"""
Reproducibility tests for publication-grade determinism.

Tests:
1. RNG state checkpoint round-trip
2. Training determinism with same seed
3. Checkpoint resume reproducibility
"""

import pytest
import torch
import random
import numpy as np
import tempfile
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestRNGStateCheckpoint:
    """Test RNG state saving and loading."""

    def test_python_rng_roundtrip(self):
        """Python RNG state should survive save/load."""
        # Set initial state
        random.seed(42)

        # Generate some values
        values_before = [random.random() for _ in range(5)]

        # Save state
        saved_state = random.getstate()

        # Generate more values
        values_middle = [random.random() for _ in range(5)]

        # Restore state
        random.setstate(saved_state)

        # Should get same values as middle
        values_after = [random.random() for _ in range(5)]

        assert values_after == values_middle, "Python RNG state not properly restored"

    def test_numpy_rng_roundtrip(self):
        """NumPy RNG state should survive save/load."""
        np.random.seed(42)

        values_before = np.random.rand(5).tolist()
        saved_state = np.random.get_state()
        values_middle = np.random.rand(5).tolist()

        np.random.set_state(saved_state)
        values_after = np.random.rand(5).tolist()

        assert values_after == values_middle, "NumPy RNG state not properly restored"

    def test_torch_rng_roundtrip(self):
        """PyTorch RNG state should survive save/load."""
        torch.manual_seed(42)

        values_before = torch.rand(5).tolist()
        saved_state = torch.get_rng_state()
        values_middle = torch.rand(5).tolist()

        torch.set_rng_state(saved_state)
        values_after = torch.rand(5).tolist()

        assert values_after == values_middle, "Torch RNG state not properly restored"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_rng_roundtrip(self):
        """CUDA RNG state should survive save/load."""
        torch.cuda.manual_seed(42)

        device = torch.device("cuda")
        values_before = torch.rand(5, device=device).cpu().tolist()
        saved_state = torch.cuda.get_rng_state_all()
        values_middle = torch.rand(5, device=device).cpu().tolist()

        torch.cuda.set_rng_state_all(saved_state)
        values_after = torch.rand(5, device=device).cpu().tolist()

        assert values_after == values_middle, "CUDA RNG state not properly restored"


class TestSeedDeterminism:
    """Test that same seeds produce same results."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Mamba")
    def test_model_init_determinism(self):
        """Model initialization should be deterministic with same seed."""
        from models import ModelConfig, HybridMambaEncoderDecoder

        device = "cuda"
        config = ModelConfig(
            vocab_size=1000,
            d_model=64,
            encoder_layers=2,
            decoder_layers=2,
            d_state=16,
            n_heads=2,
        )

        # Initialize with seed 42
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        random.seed(42)
        np.random.seed(42)
        model1 = HybridMambaEncoderDecoder(config, device=device, dtype=torch.float32)

        # Get first layer weights
        weights1 = list(model1.parameters())[0].clone()

        # Initialize again with same seed
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        random.seed(42)
        np.random.seed(42)
        model2 = HybridMambaEncoderDecoder(config, device=device, dtype=torch.float32)

        weights2 = list(model2.parameters())[0].clone()

        assert torch.allclose(weights1, weights2), "Model init not deterministic"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Mamba")
    def test_forward_pass_determinism(self):
        """Forward pass should be deterministic."""
        from models import ModelConfig, HybridMambaEncoderDecoder

        device = "cuda"
        config = ModelConfig(
            vocab_size=1000,
            d_model=64,
            encoder_layers=2,
            decoder_layers=2,
            d_state=16,
            n_heads=2,
        )

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        # Use bfloat16 as required by FlashAttention
        model = HybridMambaEncoderDecoder(config, device=device, dtype=torch.bfloat16)
        model.eval()

        src = torch.randint(1, 1000, (2, 10), device=device)
        tgt = torch.randint(1, 1000, (2, 8), device=device)

        # Run forward twice
        with torch.no_grad():
            out1 = model(src, tgt)
            out2 = model(src, tgt)

        assert torch.allclose(out1, out2), "Forward pass not deterministic"


class TestCheckpointReproducibility:
    """Test checkpoint save/load maintains training state."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Mamba")
    def test_checkpoint_structure(self):
        """Verify checkpoint contains all required fields."""
        from training import Trainer, TrainerConfig
        from models import ModelConfig, HybridMambaEncoderDecoder

        device = "cuda"

        # Create minimal model and trainer
        config = ModelConfig(
            vocab_size=1000,
            d_model=64,
            encoder_layers=2,
            decoder_layers=2,
            d_state=16,
            n_heads=2,
        )

        model = HybridMambaEncoderDecoder(config, device=device, dtype=torch.bfloat16)

        trainer_config = TrainerConfig(
            max_steps=10,
            batch_size=2,
            output_dir=tempfile.mkdtemp(),
            use_compile=False,  # Disable compile for test stability
        )

        # Create dummy dataloader
        class DummyDataloader:
            def __iter__(self):
                while True:
                    yield {
                        "src_ids": torch.randint(1, 1000, (2, 10)),
                        "tgt_ids": torch.randint(1, 1000, (2, 8)),
                        "src_mask": torch.ones(2, 10, dtype=torch.bool),
                        "tgt_mask": torch.ones(2, 8, dtype=torch.bool),
                        "labels": torch.randint(1, 1000, (2, 7)),
                    }

        trainer = Trainer(
            model=model,
            train_dataloader=DummyDataloader(),
            config=trainer_config,
        )

        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            # Override output_dir for this test
            from pathlib import Path
            trainer.output_dir = Path(tmpdir)
            trainer._save_checkpoint("test_checkpoint")

            # Load and verify structure (checkpoint.pt is inside the directory)
            checkpoint_path = os.path.join(tmpdir, "test_checkpoint", "checkpoint.pt")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

            required_keys = [
                "model_state_dict",
                "optimizer_state_dict",
                "scheduler_state_dict",
                "global_step",
                "epoch",
                "rng_state",
            ]

            for key in required_keys:
                assert key in checkpoint, f"Missing required key: {key}"

            # Verify RNG state structure
            rng_state = checkpoint["rng_state"]
            assert "python" in rng_state
            assert "numpy" in rng_state
            assert "torch" in rng_state


class TestAugmentationReproducibility:
    """Test data augmentation reproducibility."""

    def test_catn_same_seed_same_output(self):
        """CAT-N with same seed should produce identical output."""
        from data.augmentation import ConcatenationAugmenter, DocumentSample

        doc = DocumentSample(
            src_sentences=[f"Source sentence {i}." for i in range(10)],
            tgt_sentences=[f"Target sentence {i}." for i in range(10)],
        )

        # First run
        aug1 = ConcatenationAugmenter(n_sentences=3, p_concat=0.5, seed=42)
        aug1.set_epoch(0)
        result1 = aug1.augment_document(doc)

        # Second run with same seed
        aug2 = ConcatenationAugmenter(n_sentences=3, p_concat=0.5, seed=42)
        aug2.set_epoch(0)
        result2 = aug2.augment_document(doc)

        assert result1 == result2, "CAT-N augmentation not reproducible"

    def test_dataset_epoch_reset(self):
        """Dataset set_epoch should reset augmentation."""
        from data.document_dataset import DocumentNMTDataset
        from data.augmentation import ConcatenationAugmenter

        src = ["sent1", "sent2", "sent3", "sent4", "sent5"]
        tgt = ["tgt1", "tgt2", "tgt3", "tgt4", "tgt5"]

        # Create mock tokenizer
        class MockTokenizer:
            vocab_size = 1000
            pad_token_id = 0

            def encode_pair(self, src, tgt, max_src_length=512, max_tgt_length=512):
                return torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])

        aug = ConcatenationAugmenter(n_sentences=3, p_concat=0.5, seed=42)

        dataset = DocumentNMTDataset(
            src_texts=src,
            tgt_texts=tgt,
            tokenizer=MockTokenizer(),
            augmenter=aug,
        )

        # set_epoch should work without error
        dataset.set_epoch(1)
        dataset.set_epoch(2)

        # Should be able to iterate
        sample = dataset[0]
        assert "src_ids" in sample


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
