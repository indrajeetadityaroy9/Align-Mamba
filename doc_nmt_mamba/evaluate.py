#!/usr/bin/env python3
"""Evaluation script for State Capacity experiments.

Supports:
- MQAR accuracy evaluation (token and sample accuracy)
- ContraPro discourse evaluation (for application experiments)

USAGE:
    # Evaluate MQAR checkpoint
    python doc_nmt_mamba/scripts/evaluate.py checkpoint=outputs/01_mqar_cliff/model.pt

    # With specific dataset config
    python doc_nmt_mamba/scripts/evaluate.py checkpoint=<path> data.mqar.num_pairs=128
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

import hydra
from omegaconf import DictConfig

from doc_nmt_mamba.models import load_model_from_checkpoint
from doc_nmt_mamba.data import MQARDataset, MQARConfig
from doc_nmt_mamba.data.mqar import compute_mqar_accuracy


def evaluate_mqar(
    model,
    config: MQARConfig,
    num_samples: int = 1000,
    batch_size: int = 32,
    device: str = "cuda",
) -> Dict[str, float]:
    """Evaluate model on MQAR task.

    Returns:
        Dict with token_accuracy and sample_accuracy
    """
    model.eval()

    # Create evaluation dataset
    dataset = MQARDataset(
        config=config,
        num_samples=num_samples,
        split="test",
    )

    total_token_correct = 0
    total_tokens = 0
    total_sample_correct = 0
    total_samples = 0

    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch_indices = list(range(i, min(i + batch_size, len(dataset))))
            batch = [dataset[j] for j in batch_indices]

            # Stack batch
            if config.mode == "decoder_only":
                input_ids = torch.stack([b["input_ids"] for b in batch]).to(device)
                labels = torch.stack([b["labels"] for b in batch]).to(device)

                # Forward pass
                outputs = model(input_ids)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs

                # Get predictions
                predictions = logits.argmax(dim=-1)

                # Compute accuracy (only on query positions where labels != -100)
                mask = labels != -100
                token_correct = ((predictions == labels) & mask).sum().item()
                token_total = mask.sum().item()

                # Sample accuracy: all query tokens correct
                sample_correct = ((predictions == labels) | ~mask).all(dim=-1).sum().item()

            else:  # seq2seq mode
                encoder_ids = torch.stack([b["encoder_ids"] for b in batch]).to(device)
                decoder_ids = torch.stack([b["decoder_ids"] for b in batch]).to(device)
                labels = torch.stack([b["labels"] for b in batch]).to(device)

                # Forward pass
                outputs = model(encoder_ids, decoder_ids)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs

                # Get predictions
                predictions = logits.argmax(dim=-1)

                # Compute accuracy
                mask = labels != -100
                token_correct = ((predictions == labels) & mask).sum().item()
                token_total = mask.sum().item()

                # Sample accuracy
                sample_correct = ((predictions == labels) | ~mask).all(dim=-1).sum().item()

            total_token_correct += token_correct
            total_tokens += token_total
            total_sample_correct += sample_correct
            total_samples += len(batch)

    return {
        "token_accuracy": total_token_correct / max(total_tokens, 1),
        "sample_accuracy": total_sample_correct / max(total_samples, 1),
        "num_samples": total_samples,
        "num_pairs": config.num_pairs,
        "d_state": getattr(model.config, "d_state", None),
    }


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Run evaluation pipeline based on config."""
    print("=" * 60)
    print("State Capacity Evaluation")
    print("=" * 60)

    if not cfg.get("checkpoint"):
        print("No checkpoint specified. Use: python evaluate.py checkpoint=<path>")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if cfg.training.get("use_bf16", True) else torch.float32

    print(f"\nDevice: {device}")
    print(f"Checkpoint: {cfg.checkpoint}")

    # Load model
    print("\nLoading model...")
    try:
        model, model_config = load_model_from_checkpoint(cfg.checkpoint, device=device, dtype=dtype)
        model.eval()
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Loaded model: {param_count / 1e6:.1f}M parameters")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Get dataset config
    data_cfg = cfg.get("data", {})
    dataset_name = data_cfg.get("dataset_name", "mqar")

    if dataset_name == "mqar":
        # MQAR evaluation
        mqar_cfg = data_cfg.get("mqar", {})
        config = MQARConfig(
            vocab_size=mqar_cfg.get("vocab_size", 8192),
            num_pairs=mqar_cfg.get("num_pairs", 64),
            num_queries=mqar_cfg.get("num_queries", 16),
            seq_length=mqar_cfg.get("seq_length", 512),
            mode=mqar_cfg.get("mode", "seq2seq"),
        )

        print(f"\nMQAR Config:")
        print(f"  num_pairs: {config.num_pairs}")
        print(f"  num_queries: {config.num_queries}")
        print(f"  mode: {config.mode}")

        num_samples = cfg.get("eval_samples", 1000)
        batch_size = cfg.training.get("batch_size", 32)

        print(f"\nEvaluating on {num_samples} samples...")
        results = evaluate_mqar(
            model=model,
            config=config,
            num_samples=num_samples,
            batch_size=batch_size,
            device=device,
        )

        print("\n" + "=" * 60)
        print("MQAR EVALUATION RESULTS")
        print("=" * 60)
        print(f"Token Accuracy:  {results['token_accuracy']*100:.2f}%")
        print(f"Sample Accuracy: {results['sample_accuracy']*100:.2f}%")
        print(f"Samples:         {results['num_samples']}")
        print(f"Num Pairs:       {results['num_pairs']}")

        # Save results
        output_dir = Path(cfg.get("output_dir", "outputs/evaluation"))
        output_dir.mkdir(parents=True, exist_ok=True)

        import json
        results_path = output_dir / "mqar_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_path}")

    else:
        print(f"Evaluation for dataset '{dataset_name}' not implemented.")
        print("Available: mqar")


if __name__ == "__main__":
    main()
