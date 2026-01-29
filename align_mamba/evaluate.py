#!/usr/bin/env python3
"""Evaluation script for State Capacity experiments.

USAGE:
    python -m align_mamba.evaluate checkpoint=outputs/01_mqar_cliff/model.pt
    python -m align_mamba.evaluate checkpoint=<path> data.num_pairs=128
"""

import json
from pathlib import Path
from typing import Dict

import torch
import hydra
from omegaconf import DictConfig

from align_mamba.models import load_model_from_checkpoint
from align_mamba.data import MQARDataset, MQARConfig


def evaluate_mqar(
    model,
    config: MQARConfig,
    num_samples: int = 1000,
    batch_size: int = 32,
    device: str = "cuda",
    mode: str = "seq2seq",
) -> Dict[str, float]:
    """Evaluate model on MQAR task."""
    model.eval()

    dataset = MQARDataset(
        config=config,
        num_samples=num_samples,
        split="test",
        mode=mode,
    )

    total_token_correct = 0
    total_tokens = 0
    total_sample_correct = 0
    total_samples = 0

    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch_indices = list(range(i, min(i + batch_size, len(dataset))))
            batch = [dataset[j] for j in batch_indices]

            if mode == "decoder_only":
                input_ids = torch.stack([b["input_ids"] for b in batch]).to(device)
                labels = torch.stack([b["labels"] for b in batch]).to(device)

                logits = model(None, input_ids)
                predictions = logits.argmax(dim=-1)

                mask = labels != -100
                token_correct = ((predictions == labels) & mask).sum().item()
                token_total = mask.sum().item()
                sample_correct = ((predictions == labels) | ~mask).all(dim=-1).sum().item()

            else:  # seq2seq mode
                src_ids = torch.stack([b["src_ids"] for b in batch]).to(device)
                tgt_ids = torch.stack([b["tgt_ids"] for b in batch]).to(device)
                labels = torch.stack([b["labels"] for b in batch]).to(device)

                decoder_input = tgt_ids[:, :-1]
                logits = model(src_ids, decoder_input)
                predictions = logits.argmax(dim=-1)

                mask = labels != -100
                token_correct = ((predictions == labels) & mask).sum().item()
                token_total = mask.sum().item()
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
    """Run evaluation pipeline."""
    print("=" * 60)
    print("State Capacity Evaluation")
    print("=" * 60)

    if not cfg.get("checkpoint"):
        raise ValueError("No checkpoint specified. Use: python evaluate.py checkpoint=<path>")

    device = "cuda"
    dtype = torch.bfloat16

    print(f"\nCheckpoint: {cfg.checkpoint}")

    # Load model - errors propagate directly
    print("\nLoading model...")
    model, model_config = load_model_from_checkpoint(cfg.checkpoint, device=device, dtype=dtype)
    model.eval()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Loaded model: {param_count / 1e6:.1f}M parameters")

    # Get dataset config
    data_cfg = cfg.get("data", {})
    mqar_cfg = data_cfg.get("mqar", {})

    config = MQARConfig(
        vocab_size=data_cfg.get("vocab_size", mqar_cfg.get("vocab_size", 8192)),
        num_pairs=data_cfg.get("num_pairs", mqar_cfg.get("num_pairs", 64)),
        num_queries=data_cfg.get("num_queries", mqar_cfg.get("num_queries", 16)),
        seq_length=data_cfg.get("seq_length", mqar_cfg.get("seq_length", 512)),
    )
    mqar_mode = data_cfg.get("mode", mqar_cfg.get("mode", "seq2seq"))

    print(f"\nMQAR Config: num_pairs={config.num_pairs}, num_queries={config.num_queries}, mode={mqar_mode}")

    num_samples = cfg.get("eval_samples", 1000)
    batch_size = cfg.training.get("batch_size", 32)

    print(f"Evaluating on {num_samples} samples...")
    results = evaluate_mqar(
        model=model,
        config=config,
        num_samples=num_samples,
        batch_size=batch_size,
        device=device,
        mode=mqar_mode,
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Token Accuracy:  {results['token_accuracy']*100:.2f}%")
    print(f"Sample Accuracy: {results['sample_accuracy']*100:.2f}%")

    # Save results
    output_dir = Path(cfg.get("output_dir", "outputs/evaluation"))
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "mqar_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
