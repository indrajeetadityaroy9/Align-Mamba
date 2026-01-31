"""Evaluation for Align-Mamba."""

import argparse
import json
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from align_mamba.config import Config
from align_mamba.model import load_checkpoint
from align_mamba.data import MQARDataset


def evaluate(
    model: nn.Module,
    config: Config,
    *,
    num_samples: int = 1000,
    batch_size: int = 32,
    device: str = "cuda",
) -> dict:
    """Evaluate on MQAR task."""
    model.eval()
    dataset = MQARDataset(config.num_pairs, config.num_queries, num_samples, "test")

    correct, total = 0, 0
    ppl_sum, ppl_n = 0.0, 0

    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]

            src = torch.stack([b["src_ids"] for b in batch]).to(device)
            tgt = torch.stack([b["tgt_ids"] for b in batch]).to(device)
            labels = torch.stack([b["labels"] for b in batch]).to(device)
            logits = model(src, tgt[:, :-1])
            labels = labels[:, :logits.size(1)]

            preds = logits.argmax(dim=-1)
            mask = labels != -100
            correct += ((preds == labels) & mask).sum().item()
            total += mask.sum().item()

            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1),
                                  ignore_index=-100, reduction='mean')
            ppl_sum += torch.exp(loss).item()
            ppl_n += 1

    return {
        "token_accuracy": correct / total if total > 0 else 0,
        "perplexity": ppl_sum / ppl_n if ppl_n > 0 else float('inf'),
        "num_pairs": config.num_pairs,
        "d_state": config.d_state,
    }


def capacity_cliff(
    model: nn.Module,
    config: Config,
    *,
    num_samples: int = 500,
    batch_size: int = 32,
    device: str = "cuda",
) -> dict:
    """Find capacity cliff where accuracy drops."""
    d_state = config.d_state
    results = []

    for num_pairs in [32, 48, 64, 80, 96, 112, 128, 160, 192, 256]:
        dataset = MQARDataset(num_pairs, min(16, num_pairs), num_samples, "test")
        correct, total = 0, 0

        with torch.no_grad():
            for i in range(0, len(dataset), batch_size):
                batch = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
                src = torch.stack([b["src_ids"] for b in batch]).to(device)
                tgt = torch.stack([b["tgt_ids"] for b in batch]).to(device)
                labels = torch.stack([b["labels"] for b in batch]).to(device)
                logits = model(src, tgt[:, :-1])
                labels = labels[:, :logits.size(1)]
                preds = logits.argmax(dim=-1)
                mask = labels != -100
                correct += ((preds == labels) & mask).sum().item()
                total += mask.sum().item()

        acc = correct / total if total > 0 else 0
        above = num_pairs > d_state
        results.append({"num_pairs": num_pairs, "token_accuracy": acc, "above_capacity": above})
        print(f"pairs={num_pairs:3d} acc={acc:.4f} {'ABOVE' if above else 'below'}")

    cliff = None
    for r in results:
        if r["above_capacity"] and r["token_accuracy"] < 0.9:
            cliff = r["num_pairs"]
            break

    return {"results": results, "cliff_point": cliff, "d_state": d_state}


EvalMode = Literal["standard", "capacity_cliff"]


def main():
    parser = argparse.ArgumentParser(prog="align-eval")
    parser.add_argument("checkpoint", help="Path to checkpoint directory")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["standard", "capacity_cliff"],
        default="standard",
        help="Evaluation mode: 'standard' or 'capacity_cliff'",
    )
    args = parser.parse_args()

    model, config = load_checkpoint(args.checkpoint, device="cuda", dtype=torch.bfloat16)

    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.mode == "capacity_cliff":
        print(f"\nCapacity cliff eval (d_state={config.d_state})")
        results = capacity_cliff(model, config)
        with open(out / "capacity_cliff.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"Cliff at pairs={results['cliff_point']}")
    else:
        results = evaluate(model, config)
        print(f"acc={results['token_accuracy']:.4f} ppl={results['perplexity']:.2f}")
        with open(out / "results.json", "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
