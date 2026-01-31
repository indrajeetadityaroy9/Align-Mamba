"""MQAR evaluation functions."""

from typing import Dict

import torch

from align_mamba.data.mqar import MQARDataset, MQARConfig
from align_mamba.evaluation import BatchMetrics, compute_batch_metrics, compute_perplexity


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

    accumulated = BatchMetrics(0, 0, 0, 0)
    total_perplexity_sum = 0.0
    perplexity_batches = 0

    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]

            if mode == "decoder_only":
                input_ids = torch.stack([b["input_ids"] for b in batch]).to(device)
                labels = torch.stack([b["labels"] for b in batch]).to(device)
                logits = model(None, input_ids)
            else:
                src_ids = torch.stack([b["src_ids"] for b in batch]).to(device)
                tgt_ids = torch.stack([b["tgt_ids"] for b in batch]).to(device)
                labels = torch.stack([b["labels"] for b in batch]).to(device)
                logits = model(src_ids, tgt_ids[:, :-1])

            predictions = logits.argmax(dim=-1)
            mask = labels != -100

            batch_metrics = compute_batch_metrics(predictions, labels, mask)
            accumulated = accumulated + batch_metrics

            total_perplexity_sum += compute_perplexity(logits, labels)
            perplexity_batches += 1

    return {
        "token_accuracy": accumulated.token_accuracy,
        "sample_accuracy": accumulated.sample_accuracy,
        "perplexity": total_perplexity_sum / perplexity_batches,
        "num_samples": accumulated.sample_total,
        "num_pairs": config.num_pairs,
        "d_state": model.config.d_state,
        "hybrid_positions": sorted(list(model.config.hybrid_positions)) if model.config.hybrid_positions else None,
    }
