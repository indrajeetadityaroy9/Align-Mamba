"""Evaluation entry point."""

import json
from pathlib import Path

import torch
import hydra
from omegaconf import DictConfig

from align_mamba.models.encoder_decoder import load_checkpoint
from align_mamba.data.mqar import MQARConfig
from align_mamba.evaluate import evaluate_mqar


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Evaluation entry point."""
    model, _ = load_checkpoint(cfg.checkpoint, device="cuda", dtype=torch.bfloat16)

    config = MQARConfig(
        num_pairs=cfg.data.num_pairs,
        num_queries=cfg.data.num_queries,
    )

    results = evaluate_mqar(
        model=model,
        config=config,
        num_samples=cfg.eval_samples,
        batch_size=cfg.training.batch_size,
        device="cuda",
        mode=cfg.data.mode,
    )

    print(f"token_acc={results['token_accuracy']:.4f} sample_acc={results['sample_accuracy']:.4f} ppl={results['perplexity']:.2f}")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
