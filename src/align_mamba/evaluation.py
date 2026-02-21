# Evaluation utilities.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from align_mamba.config import Config
from align_mamba.data import MQARDataset, collate

_DEFAULT_SWEEP_PAIRS = [32, 48, 64, 80, 96, 112, 128, 160, 192, 256]
_DEFAULT_CLIFF_THRESHOLD = 0.9


def evaluate(
    model: nn.Module,
    config: Config,
    *,
    num_pairs: int,
    num_queries: int,
    num_samples: int = 1000,
    batch_size: int = 32,
) -> dict:
    # Run standard MQAR evaluation.
    model.eval()
    device = next(model.parameters()).device

    dataset = MQARDataset(num_pairs, num_queries, num_samples, "test")
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate, pin_memory=True)

    correct, total = 0, 0
    loss_sum, loss_tokens = 0.0, 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            logits = model(batch["src_ids"], batch["tgt_ids"][:, :-1])
            labels = batch["labels"][:, :logits.size(1)]

            preds = logits.argmax(dim=-1)
            mask = labels != -100
            correct += ((preds == labels) & mask).sum().item()
            total += mask.sum().item()

            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1),
                                  ignore_index=-100, reduction='sum')
            loss_sum += loss.item()
            loss_tokens += mask.sum().item()

    return {
        "token_accuracy": correct / total,
        "perplexity": math.exp(loss_sum / loss_tokens),
        "num_pairs": num_pairs,
        "d_state": config.d_state,
    }


def capacity_cliff(
    model: nn.Module,
    config: Config,
    *,
    num_samples: int = 500,
    batch_size: int = 32,
    sweep_pairs: list = _DEFAULT_SWEEP_PAIRS,
    cliff_threshold: float = _DEFAULT_CLIFF_THRESHOLD,
) -> dict:
    # Sweep `num_pairs` and report the first post-capacity collapse point.
    d_state = config.d_state
    results = []

    for np_ in sweep_pairs:
        r = evaluate(model, config, num_pairs=np_,
                     num_queries=min(config.num_queries, np_),
                     num_samples=num_samples, batch_size=batch_size)
        results.append({
            "num_pairs": np_,
            "token_accuracy": r["token_accuracy"],
            "above_capacity": np_ > d_state,
        })

    cliff = None
    for r in results:
        if r["above_capacity"] and r["token_accuracy"] < cliff_threshold:
            cliff = r["num_pairs"]
            break

    return {"results": results, "cliff_point": cliff, "d_state": d_state}


def run_evaluation(
    model: nn.Module,
    config: Config,
    eval_cfg: dict,
) -> dict:
    # Dispatch evaluation by `eval_cfg["mode"]`.
    mode = eval_cfg["mode"]
    kwargs = {}
    for key in ("num_samples", "batch_size"):
        if key in eval_cfg:
            kwargs[key] = eval_cfg[key]

    if mode == "capacity_cliff":
        for key in ("sweep_pairs", "cliff_threshold"):
            if key in eval_cfg:
                kwargs[key] = eval_cfg[key]
        result = capacity_cliff(model, config, **kwargs)
    else:
        result = evaluate(
            model, config,
            num_pairs=config.num_pairs,
            num_queries=config.num_queries,
            **kwargs,
        )

    result["mode"] = mode
    return result
