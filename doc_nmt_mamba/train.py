#!/usr/bin/env python3
"""Training script for Hybrid Mamba-Attention NMT. Supports single/multi-GPU via DDP/FSDP."""

import os
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

import hydra
from omegaconf import DictConfig, OmegaConf

from doc_nmt_mamba.models import ModelConfig, HybridMambaEncoderDecoder
from doc_nmt_mamba.data import (
    CustomBPETokenizer,
    create_tokenizer,
    create_dataset,
    create_collator,
    DocumentConcatenationAugmenter,
    MQARDataset,
    MQARConfig,
    MQARCollator,
)
from doc_nmt_mamba.training import NMTTrainer, NMTTrainerConfig, setup_distributed


def setup_environment():
    """Configure H100 optimizations: TF32, cudnn, NCCL for NVLink."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("NCCL_IB_DISABLE", "0")
    os.environ.setdefault("NCCL_P2P_LEVEL", "NVL")


def validate_config(cfg: DictConfig) -> None:
    """Catch common config errors early."""
    # MQAR mode consistency: warn if both levels defined differently
    if cfg.data.get("dataset_name") == "mqar" or cfg.data.get("dataset_type") == "synthetic":
        mode_toplevel = cfg.data.get("mode")
        mode_nested = cfg.data.get("mqar", {}).get("mode")
        if mode_toplevel and mode_nested and mode_toplevel != mode_nested:
            raise ValueError(
                f"Conflicting MQAR modes: data.mode={mode_toplevel} vs data.mqar.mode={mode_nested}. "
                f"Use only one."
            )

        # d_state consistency warning for capacity experiments
        model_d_state = cfg.model.get("d_state")
        data_d_state = cfg.data.get("mqar", {}).get("d_state")
        if model_d_state and data_d_state and model_d_state != data_d_state:
            print(f"WARNING: model.d_state={model_d_state} != data.mqar.d_state={data_d_state}")

        # num_pairs vs d_state sanity check
        num_pairs = cfg.data.get("mqar", {}).get("num_pairs", cfg.data.get("num_pairs"))
        d_state = model_d_state or data_d_state or 64
        if num_pairs and num_pairs > d_state * 2:
            print(f"INFO: num_pairs={num_pairs} > 2*d_state={2*d_state} - testing capacity cliff")


def create_model(cfg: DictConfig, device: str, dtype: torch.dtype) -> HybridMambaEncoderDecoder:
    """Instantiate model from Hydra config."""
    hybrid_positions = cfg.model.get("hybrid_positions")
    if hybrid_positions is not None:
        hybrid_positions = list(hybrid_positions)

    mimetic_init = cfg.model.get("mimetic_init", False)

    model_cfg = ModelConfig(
        vocab_size=cfg.model.vocab_size,
        d_model=cfg.model.d_model,
        encoder_layers=cfg.model.encoder_layers,
        decoder_layers=cfg.model.decoder_layers,
        d_state=cfg.model.d_state,
        n_heads=cfg.model.n_heads,
        attention_ratio=cfg.model.get("attention_ratio", 0.125),
        dropout=cfg.model.get("dropout", 0.1),
        max_seq_len=cfg.model.get("max_seq_len", 8192),
        hybrid_positions=hybrid_positions,
        mimetic_init=mimetic_init,
    )

    model = HybridMambaEncoderDecoder(config=model_cfg, device=device, dtype=dtype)

    print(f"Created model with {model.num_parameters() / 1e6:.1f}M parameters")
    print(f"Encoder: {model.encoder.get_layer_counts()}")
    print(f"Decoder: {model.decoder.get_layer_counts()}")
    print(f"Decoder hybrid_positions: {sorted(list(model.decoder.hybrid_positions))}")
    if mimetic_init:
        print("Mimetic initialization: ENABLED (A_log zeroed)")

    return model


def worker_init_fn(worker_id: int):
    """Derive unique per-worker seed for reproducible data loading."""
    worker_seed = torch.initial_seed() % (2**32) + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloaders(cfg: DictConfig, tokenizer, dist_info: dict):
    """Create train/val dataloaders with distributed sampling if multi-GPU."""
    dataset_name = cfg.data.get("dataset_name", "opus_books")
    dataset_type = cfg.data.get("dataset_type", "nmt")

    if dataset_type == "synthetic" or dataset_name == "mqar":
        mqar_cfg = cfg.data.get("mqar", {})
        mqar_config = MQARConfig(
            d_state=mqar_cfg.get("d_state", 64),
            num_pairs=mqar_cfg.get("num_pairs", 64),
            num_queries=mqar_cfg.get("num_queries", 16),
            vocab_size=mqar_cfg.get("vocab_size", 8192),
            seq_length=mqar_cfg.get("seq_length", 512),
            pad_token_id=mqar_cfg.get("pad_token_id", 0),
            bos_token_id=mqar_cfg.get("bos_token_id", 1),
            eos_token_id=mqar_cfg.get("eos_token_id", 2),
            kv_sep_token_id=mqar_cfg.get("kv_sep_token_id", 3),
            query_token_id=mqar_cfg.get("query_token_id", 4),
            key_token_start=mqar_cfg.get("key_token_start", 10),
            key_token_end=mqar_cfg.get("key_token_end", 4096),
            value_token_start=mqar_cfg.get("value_token_start", 4096),
            value_token_end=mqar_cfg.get("value_token_end", 8192),
        )

        # Check both cfg.data.mode (experiment override) and cfg.data.mqar.mode (nested)
        # decoder_only: tests TC0 state capacity; seq2seq: tests NC1 cross-attention retrieval
        mqar_mode = cfg.data.get("mode", mqar_cfg.get("mode", "decoder_only"))
        curriculum_cfg = cfg.data.get("curriculum", {})
        num_samples = curriculum_cfg.get("samples_per_stage", 10000)

        train_dataset = MQARDataset(config=mqar_config, num_samples=num_samples, split="train", mode=mqar_mode)
        val_dataset = MQARDataset(config=mqar_config, num_samples=num_samples // 10, split="validation", mode=mqar_mode)
        collator = MQARCollator(pad_token_id=mqar_config.pad_token_id, mode=mqar_mode)

        if dist_info.get("is_main", True):
            print(f"MQAR mode: {mqar_mode}")
    else:
        augmenter = DocumentConcatenationAugmenter(n_sentences=cfg.data.cat_n, p_concat=cfg.data.p_concat)

        train_dataset = create_dataset(
            dataset_name=dataset_name, split="train", tokenizer=tokenizer, augmenter=augmenter,
            max_src_length=cfg.data.max_src_length, max_tgt_length=cfg.data.max_tgt_length,
        )
        val_dataset = create_dataset(
            dataset_name=dataset_name, split="validation", tokenizer=tokenizer, augmenter=None,
            max_src_length=cfg.data.max_src_length, max_tgt_length=cfg.data.max_tgt_length,
        )
        collator = create_collator(mode=cfg.data.collator_mode, pad_token_id=tokenizer.pad_token_id)

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
    pin_memory = cfg.data.get("pin_memory", True) and num_workers > 0
    persistent_workers = cfg.data.get("persistent_workers", True) and num_workers > 0

    loader_kwargs = {
        "batch_size": cfg.training.batch_size,
        "num_workers": num_workers,
        "collate_fn": collator,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "drop_last": cfg.data.get("drop_last", False),
        "worker_init_fn": worker_init_fn if num_workers > 0 else None,
    }

    if num_workers > 0 and cfg.data.get("prefetch_factor"):
        loader_kwargs["prefetch_factor"] = cfg.data.prefetch_factor

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
        print("Document-Level NMT with Hybrid Mamba-Attention")
        print("=" * 60)
        validate_config(cfg)
        print(OmegaConf.to_yaml(cfg))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training")

    device = dist_info["device"]
    dtype = torch.bfloat16 if cfg.training.use_bf16 else torch.float32

    if is_main:
        print(f"\nDevice: {torch.cuda.get_device_name(device.index if device.index else 0)}")
        print(f"Dtype: {dtype}, World Size: {dist_info['world_size']}")

    dataset_type = cfg.data.get("dataset_type", "nmt")
    is_mqar = dataset_type == "synthetic" or cfg.data.get("dataset_name") == "mqar"

    if is_mqar:
        if is_main:
            print("\nMQAR synthetic task - using internal vocabulary")
        mqar_vocab_size = cfg.data.get("mqar", {}).get("vocab_size", 8192)
        cfg.model.vocab_size = mqar_vocab_size
        tokenizer = None
    else:
        if is_main:
            print("\nLoading tokenizer...")
        tokenizer_path = cfg.data.get("tokenizer_path", "data/tokenizer/tokenizer.json")
        tokenizer = create_tokenizer(tokenizer_type="custom", tokenizer_path=tokenizer_path, max_length=cfg.data.max_src_length)
        if is_main:
            print(f"Tokenizer: Custom 32K BPE, Vocab size: {tokenizer.vocab_size}")
        cfg.model.vocab_size = tokenizer.vocab_size

    if is_main:
        print("\nCreating model...")
    model = create_model(cfg, str(device), dtype)

    if is_main:
        print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(cfg, tokenizer, dist_info)

    trainer_config = NMTTrainerConfig(
        seed=cfg.project.get("seed", 42),
        max_steps=cfg.training.max_steps,
        batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        max_grad_norm=cfg.training.max_grad_norm,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        betas=tuple(cfg.training.get("betas", [0.9, 0.95])),
        eps=cfg.training.get("eps", 1e-8),
        warmup_steps=cfg.training.warmup_steps,
        scheduler_type=cfg.training.scheduler_type,
        min_lr=cfg.training.min_lr,
        use_bf16=cfg.training.use_bf16,
        use_compile=cfg.training.use_compile,
        compile_mode=cfg.training.compile_mode,
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        output_dir=cfg.training.output_dir,
        log_steps=cfg.training.log_steps,
        eval_steps=cfg.training.eval_steps,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        distributed_strategy=cfg.training.get("distributed_strategy", "ddp"),
        static_graph=cfg.training.get("static_graph", True),
        fsdp_sharding=cfg.training.get("fsdp_sharding", "full_shard"),
        fsdp_cpu_offload=cfg.training.get("fsdp_cpu_offload", False),
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
