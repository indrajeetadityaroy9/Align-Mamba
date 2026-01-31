"""Data loader factories."""

import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from omegaconf import DictConfig

from align_mamba.data.mqar import MQARDataset, MQARConfig, MQARCollator
from align_mamba.config import PAD_TOKEN_ID


def worker_init_fn(worker_id: int):
    """Derive unique per-worker seed for reproducible data loading."""
    worker_seed = torch.initial_seed() % (2**32) + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloaders(
    cfg: DictConfig,
    world_size: int,
    rank: int,
) -> Tuple[DataLoader, DataLoader]:
    """Create train/val dataloaders with distributed sampling if multi-GPU."""
    mqar_config = MQARConfig(
        num_pairs=cfg.data.num_pairs,
        num_queries=cfg.data.num_queries,
    )

    train_dataset = MQARDataset(
        config=mqar_config,
        num_samples=cfg.data.num_samples,
        split="train",
        mode=cfg.data.mode,
    )
    val_dataset = MQARDataset(
        config=mqar_config,
        num_samples=cfg.data.num_samples // 10,
        split="validation",
        mode=cfg.data.mode,
    )
    collator = MQARCollator(pad_token_id=PAD_TOKEN_ID, mode=cfg.data.mode)

    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        shuffle_train = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle_train = True

    loader_kwargs = {
        "batch_size": cfg.training.batch_size,
        "num_workers": cfg.data.num_workers,
        "collate_fn": collator,
        "pin_memory": cfg.data.num_workers > 0,
        "persistent_workers": cfg.data.num_workers > 0,
        "drop_last": world_size > 1,
        "worker_init_fn": worker_init_fn if cfg.data.num_workers > 0 else None,
    }

    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        shuffle=shuffle_train if train_sampler is None else False,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        shuffle=False,
        **loader_kwargs,
    )

    return train_loader, val_loader
