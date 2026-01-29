#!/usr/bin/env python3
"""Training script for Hybrid Mamba-Attention State Capacity experiments."""

import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import hydra
from omegaconf import DictConfig, OmegaConf

from align_mamba.models import ModelConfig, HybridMambaEncoderDecoder
from align_mamba.data import MQARDataset, MQARConfig, MQARCollator
from align_mamba.training import NMTTrainer, NMTTrainerConfig, setup_distributed
from align_mamba.constants import PAD_TOKEN_ID, MQAR_VOCAB_SIZE


def setup_environment():
    """Configure H100 optimizations: TF32, cudnn, NCCL for NVLink."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("NCCL_IB_DISABLE", "0")
    os.environ.setdefault("NCCL_P2P_LEVEL", "NVL")


def create_model(cfg: DictConfig, device: str, dtype: torch.dtype) -> HybridMambaEncoderDecoder:
    """Instantiate model from Hydra config."""
    hybrid_positions = cfg.model.get("hybrid_positions")
    if hybrid_positions is not None:
        hybrid_positions = list(hybrid_positions)

    model_cfg = ModelConfig(
        vocab_size=cfg.model.get("vocab_size", MQAR_VOCAB_SIZE),
        d_model=cfg.model.d_model,
        encoder_layers=cfg.model.encoder_layers,
        decoder_layers=cfg.model.decoder_layers,
        d_state=cfg.model.d_state,
        n_heads=cfg.model.n_heads,
        hybrid_positions=hybrid_positions,
    )

    model = HybridMambaEncoderDecoder(config=model_cfg, device=device, dtype=dtype)

    print(f"Created model with {model.num_parameters() / 1e6:.1f}M parameters")
    print(f"Encoder: {model.encoder.get_layer_counts()}")
    print(f"Decoder: {model.decoder.get_layer_counts()}")
    print(f"Decoder hybrid_positions: {sorted(list(model.decoder.hybrid_positions))}")

    return model


def worker_init_fn(worker_id: int):
    """Derive unique per-worker seed for reproducible data loading."""
    worker_seed = torch.initial_seed() % (2**32) + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloaders(cfg: DictConfig, dist_info: dict):
    """Create train/val dataloaders with distributed sampling if multi-GPU."""
    mqar_config = MQARConfig(
        num_pairs=cfg.data.get("num_pairs", 64),
        num_queries=cfg.data.get("num_queries", 16),
    )

    mqar_mode = cfg.data.get("mode", "seq2seq")
    num_samples = cfg.data.get("num_samples", 10000)

    train_dataset = MQARDataset(config=mqar_config, num_samples=num_samples, split="train", mode=mqar_mode)
    val_dataset = MQARDataset(config=mqar_config, num_samples=num_samples // 10, split="validation", mode=mqar_mode)
    collator = MQARCollator(pad_token_id=PAD_TOKEN_ID, mode=mqar_mode)

    if dist_info.get("is_main", True):
        print(f"MQAR Dataset: num_pairs={mqar_config.num_pairs}, mode={mqar_mode}")

    world_size = dist_info.get("world_size", 1)
    rank = dist_info.get("rank", 0)

    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        shuffle_train = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle_train = True

    num_workers = cfg.data.get("num_workers", 8)
    pin_memory = num_workers > 0
    persistent_workers = num_workers > 0

    loader_kwargs = {
        "batch_size": cfg.training.batch_size,
        "num_workers": num_workers,
        "collate_fn": collator,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "drop_last": False,
        "worker_init_fn": worker_init_fn if num_workers > 0 else None,
    }

    train_loader = DataLoader(train_dataset, sampler=train_sampler, shuffle=shuffle_train if train_sampler is None else False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, shuffle=False, **loader_kwargs)

    if dist_info.get("is_main", True):
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        if world_size > 1:
            print(f"Distributed: {world_size} GPUs, {len(train_dataset) // world_size} samples/GPU")

    return train_loader, val_loader


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Entry point: setup, model creation, training loop."""
    setup_environment()
    dist_info = setup_distributed()
    is_main = dist_info.get("is_main", True)

    if is_main:
        print("=" * 60)
        print("Align-Mamba: State Capacity Experiments")
        print("=" * 60)
        print(OmegaConf.to_yaml(cfg))

    device = dist_info["device"]
    dtype = torch.bfloat16

    if is_main:
        print(f"\nDevice: {torch.cuda.get_device_name(device.index if device.index else 0)}")
        print(f"Dtype: {dtype}, World Size: {dist_info['world_size']}")
        print("\nMQAR synthetic task - using internal vocabulary")

    if is_main:
        print("\nCreating model...")
    model = create_model(cfg, str(device), dtype)

    if is_main:
        print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(cfg, dist_info)

    trainer_config = NMTTrainerConfig(
        seed=cfg.project.get("seed", 42),
        max_steps=cfg.training.max_steps,
        batch_size=cfg.training.batch_size,
        learning_rate=cfg.training.learning_rate,
        label_smoothing=cfg.training.get("label_smoothing"),
        output_dir=cfg.training.output_dir,
    )

    if is_main:
        print("\nInitializing trainer...")
    trainer = NMTTrainer(model=model, train_dataloader=train_loader, config=trainer_config, eval_dataloader=val_loader)

    if cfg.training.get("resume_from"):
        trainer.load_checkpoint(cfg.training.resume_from)

    if is_main:
        print("\nStarting training...")
    trainer.train()

    if is_main:
        print("\nTraining complete!")


if __name__ == "__main__":
    main()
