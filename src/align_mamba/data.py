# MQAR data pipeline.
import os
import random
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from align_mamba.config import (
    Config, PAD_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID, SEP_TOKEN_ID,
    KEY_TOKEN_START, KEY_TOKEN_END, VALUE_TOKEN_START, VALUE_TOKEN_END,
)

Split = Literal["train", "validation", "test"]


class MQARDataset(Dataset):
    # Multi-Query Associative Recall dataset.
    SPLIT_SEEDS: dict[str, int] = {"train": 42, "validation": 1042, "test": 2042}

    def __init__(self, num_pairs: int, num_queries: int, num_samples: int, split: Split):
        self.num_pairs = num_pairs
        self.num_queries = min(num_queries, num_pairs)

        self._rng = random.Random(self.SPLIT_SEEDS[split])
        self._samples = [self._gen() for _ in range(num_samples)]

    def _gen(self) -> dict[str, torch.Tensor]:
        keys = self._rng.sample(range(KEY_TOKEN_START, KEY_TOKEN_END), self.num_pairs)
        values = [self._rng.choice(range(VALUE_TOKEN_START, VALUE_TOKEN_END)) for _ in range(self.num_pairs)]
        kv = dict(zip(keys, values))
        qkeys = self._rng.sample(keys, self.num_queries)

        src = [BOS_TOKEN_ID]
        for k, v in zip(keys, values):
            src.extend([k, SEP_TOKEN_ID, v])
        src.append(EOS_TOKEN_ID)

        tgt = [BOS_TOKEN_ID] + qkeys + [EOS_TOKEN_ID]
        labels = [-100] + [kv[k] for k in qkeys]

        return {
            'src_ids': torch.tensor(src, dtype=torch.long),
            'tgt_ids': torch.tensor(tgt, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        return self._samples[idx]


def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    # Pad variable-length sequences in a batch.
    def pad(seqs, val):
        max_len = max(len(s) for s in seqs)
        return torch.stack([F.pad(s, (0, max_len - len(s)), value=val) for s in seqs])

    src = pad([b['src_ids'] for b in batch], PAD_TOKEN_ID)
    tgt = pad([b['tgt_ids'] for b in batch], PAD_TOKEN_ID)
    labels = pad([b['labels'] for b in batch], -100)
    return {'src_ids': src, 'tgt_ids': tgt, 'labels': labels}


def create_dataloaders(config: Config) -> tuple[DataLoader, DataLoader]:
    # Build train/validation dataloaders.
    train = MQARDataset(config.num_pairs, config.num_queries, config.num_samples, "train")
    val_samples = int(config.num_samples * config.val_ratio)
    val = MQARDataset(config.num_pairs, config.num_queries, val_samples, "validation")

    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": min(8, os.cpu_count()),
        "collate_fn": collate,
        "pin_memory": True,
        "persistent_workers": True,
        "drop_last": True,
        "worker_init_fn": lambda w: (np.random.seed(torch.initial_seed() % 2**32 + w),
                                      random.seed(torch.initial_seed() % 2**32 + w)),
    }

    return (
        DataLoader(train, shuffle=True, **loader_kwargs),
        DataLoader(val, shuffle=False, **loader_kwargs),
    )
