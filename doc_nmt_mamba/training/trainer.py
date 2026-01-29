"""NMT Trainer with H100/BF16 optimization and distributed training support."""

import os
import time
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Iterator
import math
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast

from .objectives import create_loss_fn, create_scheduler
from .distributed import (
    DistributedConfig,
    setup_distributed,
    cleanup_distributed,
    wrap_model_distributed,
    print_distributed_info,
    all_reduce_mean,
    barrier,
)
from .hardware import (
    detect_hardware,
    print_hardware_info,
    setup_h100_optimizations,
    setup_nccl_optimizations,
    CUDAMemoryManager,
    get_optimal_worker_count,
    print_h100_optimization_status,
)
from ..models.utils import get_unwrapped_model, split_params_for_weight_decay
from ..models.align_mamba import CurriculumDropout


def adaptive_gradient_clip_(
    parameters: Iterator[nn.Parameter],
    clip_factor: float = 0.01,
    eps: float = 1e-3,
) -> None:
    """
    Adaptive Gradient Clipping (AGC) from NFNet paper (Brock et al., 2021).

    Clips gradients based on the ratio of weight norm to gradient norm,
    which adapts per-parameter without manual tuning.

    Args:
        parameters: Model parameters to clip
        clip_factor: Maximum ||grad|| / ||weight|| ratio (0.01 works universally)
        eps: Small constant for numerical stability
    """
    for p in parameters:
        if p.grad is None:
            continue
        param_norm = p.data.norm(p=2).clamp(min=eps)
        grad_norm = p.grad.data.norm(p=2)
        max_norm = param_norm * clip_factor
        if grad_norm > max_norm:
            p.grad.data.mul_(max_norm / (grad_norm + eps))


def compute_adaptive_smoothing(vocab_size: int, base_smoothing: float = 0.1) -> float:
    """
    Compute label smoothing scaled by vocabulary size.

    Larger vocabularies have more uncertainty, so need more smoothing.
    Formula: smoothing = base_smoothing * log2(vocab_size) / log2(32768)

    Args:
        vocab_size: Model vocabulary size
        base_smoothing: Base smoothing value (0.1 standard for NMT)

    Returns:
        Scaled smoothing value
    """
    return base_smoothing * math.log2(vocab_size) / math.log2(32768)


@dataclass
class NMTTrainerConfig:
    seed: int = 42  # Random seed for reproducibility
    max_steps: int = 100000
    batch_size: int = 64
    gradient_accumulation_steps: int = 1
    # Optimizer settings
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    # Gradient clipping
    use_agc: bool = True  # Adaptive Gradient Clipping (Brock et al., 2021)
    agc_clip_factor: float = 0.01
    max_grad_norm: float = 1.0  # Fallback if use_agc=False
    # Learning rate schedule (adaptive warmup)
    warmup_ratio: float = 0.05  # Warmup as fraction of max_steps
    min_warmup_steps: int = 100  # Floor for short training runs
    scheduler_type: str = "cosine"
    min_lr: float = 1e-6
    # Label smoothing (adaptive if None)
    label_smoothing: Optional[float] = None  # None = adaptive based on vocab_size
    # Precision and compilation
    use_bf16: bool = True
    use_compile: bool = True
    compile_mode: str = "max-autotune"
    # Checkpointing
    save_steps: int = 5000
    save_total_limit: int = 3
    output_dir: str = "outputs"
    log_steps: int = 100
    eval_steps: int = 1000
    gradient_checkpointing: bool = True
    # Hardware optimizations
    tf32_matmul: bool = True
    cudnn_benchmark: bool = True
    use_fused_optimizer: bool = True  # fused=True runs optimizer on GPU, ~5-10% speedup
    # Distributed training
    bucket_cap_mb: int = 512
    distributed_strategy: str = "ddp"
    static_graph: bool = True
    fsdp_sharding: str = "full_shard"
    fsdp_cpu_offload: bool = False


class NMTTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        config: Optional[NMTTrainerConfig] = None,
        eval_dataloader: Optional[DataLoader] = None,
        eval_fn: Optional[Callable] = None,
    ):
        self.config = config or NMTTrainerConfig()
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.eval_fn = eval_fn

        self.dist_info = setup_distributed()
        self.device = self.dist_info["device"]
        self.is_main = self.dist_info["is_main"]
        self.world_size = self.dist_info["world_size"]

        print_distributed_info(self.dist_info)
        self._setup_reproducibility()
        self._setup_h100_optimizations()

        if self.config.gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()

        # IMPORTANT: torch.compile BEFORE distributed wrapping for optimal performance
        # Compiling after DDP can cause graph breaks and reduce speedup
        if self.config.use_compile:
            self.model = torch.compile(self.model, mode=self.config.compile_mode)

        if self.world_size > 1:
            dist_config = DistributedConfig(
                strategy=self.config.distributed_strategy,
                static_graph=self.config.static_graph,
                sharding_strategy=self.config.fsdp_sharding,
                cpu_offload=self.config.fsdp_cpu_offload,
            )
            self.model = wrap_model_distributed(
                self.model,
                dist_config,
                self.device,
                use_bf16=self.config.use_bf16,
            )
        else:
            self.model = self.model.to(self.device)

        self.optimizer = self._create_optimizer()

        # Adaptive warmup: ratio of max_steps with minimum floor
        adaptive_warmup = max(
            int(self.config.max_steps * self.config.warmup_ratio),
            self.config.min_warmup_steps,
        )
        if self.is_main:
            print(f"Warmup steps: {adaptive_warmup} ({self.config.warmup_ratio*100:.0f}% of {self.config.max_steps})")

        self.scheduler = create_scheduler(
            scheduler_type=self.config.scheduler_type,
            optimizer=self.optimizer,
            warmup_steps=adaptive_warmup,
            max_steps=self.config.max_steps,
            min_lr=self.config.min_lr,
        )

        # Adaptive label smoothing based on vocab size
        if self.config.label_smoothing is not None:
            smoothing = self.config.label_smoothing
        else:
            # Get vocab_size from model config
            vocab_size = getattr(getattr(model, 'config', None), 'vocab_size', 32768)
            smoothing = compute_adaptive_smoothing(vocab_size)
            if self.is_main:
                print(f"Adaptive label smoothing: {smoothing:.4f} (vocab_size={vocab_size})")

        self.loss_fn = create_loss_fn(
            loss_type="label_smoothing",
            smoothing=smoothing,
            ignore_index=-100,
        )

        self.global_step = 0
        self.epoch = 0
        self.best_metric = float("inf")

        self.output_dir = Path(self.config.output_dir)
        if self.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        barrier()

    def _setup_h100_optimizations(self):
        if self.is_main:
            self.hardware_info = detect_hardware()
            print_hardware_info(self.hardware_info)
        else:
            self.hardware_info = detect_hardware()

        setup_h100_optimizations()

        if self.world_size > 1:
            setup_nccl_optimizations(
                nvlink_available=self.hardware_info.nvlink_available,
                use_infiniband=False,
            )

        if self.is_main:
            print_h100_optimization_status()

    def _setup_reproducibility(self):
        seed = self.config.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if self.is_main:
            print(f"Random seed: {seed}")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        optimizer_groups = split_params_for_weight_decay(
            self.model, self.config.weight_decay
        )

        use_fused = (
            self.config.use_fused_optimizer
            and torch.cuda.is_available()
            and self.device.type == "cuda"
        )

        if self.is_main:
            print(f"Using AdamW optimizer (fused={use_fused})")

        return torch.optim.AdamW(
            optimizer_groups,
            lr=self.config.learning_rate,
            betas=self.config.betas,
            eps=self.config.eps,
            fused=use_fused,
        )

    @classmethod
    def create_dataloader(
        cls,
        dataset,
        batch_size: int,
        is_train: bool = True,
        world_size: int = 1,
        rank: int = 0,
        collate_fn=None,
        drop_last: bool = True,
    ) -> DataLoader:
        num_workers = get_optimal_worker_count(world_size)

        sampler = None
        shuffle = is_train
        if world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
            )
            shuffle = False

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=4 if num_workers > 0 else None,
            drop_last=drop_last,
        )

    def train(self):
        self.model.train()
        data_iter = iter(self.train_dataloader)
        # GPU tensor avoids CPU sync every step - only sync at log_steps
        accumulated_loss = torch.tensor(0.0, device=self.device)
        step_start_time = time.time()

        if hasattr(self.train_dataloader, 'sampler') and hasattr(self.train_dataloader.sampler, 'set_epoch'):
            self.train_dataloader.sampler.set_epoch(self.epoch)

        if hasattr(self.train_dataloader, 'dataset') and hasattr(self.train_dataloader.dataset, 'set_epoch'):
            self.train_dataloader.dataset.set_epoch(self.epoch)

        while self.global_step < self.config.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                self.epoch += 1
                if hasattr(self.train_dataloader, 'sampler') and hasattr(self.train_dataloader.sampler, 'set_epoch'):
                    self.train_dataloader.sampler.set_epoch(self.epoch)
                if hasattr(self.train_dataloader, 'dataset') and hasattr(self.train_dataloader.dataset, 'set_epoch'):
                    self.train_dataloader.dataset.set_epoch(self.epoch)
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)

            batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            with autocast(dtype=torch.bfloat16, enabled=self.config.use_bf16):
                loss = self._training_step(batch)
                loss = loss / self.config.gradient_accumulation_steps

            # Check before backward to avoid corrupting gradients (LOCK-4: enhanced diagnostics)
            if torch.isnan(loss) or torch.isinf(loss):
                if self.is_main:
                    print(f"WARNING: NaN/Inf loss at step {self.global_step}, skipping")
                    print(f"  Batch keys: {list(batch.keys())}")
                    if "labels" in batch:
                        labels = batch["labels"]
                        valid_labels = labels[labels != -100]
                        if valid_labels.numel() > 0:
                            print(f"  Labels range: [{valid_labels.min().item()}, {valid_labels.max().item()}]")
                        else:
                            print("  Labels: all ignored (-100)")
                    if "input_ids" in batch:
                        print(f"  Input shape: {batch['input_ids'].shape}")
                    elif "src_ids" in batch:
                        print(f"  Src shape: {batch['src_ids'].shape}, Tgt shape: {batch['tgt_ids'].shape}")
                self.optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()
            accumulated_loss = accumulated_loss + loss.detach()

            if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping: AGC (adaptive) or fixed norm
                if self.config.use_agc:
                    adaptive_gradient_clip_(
                        self.model.parameters(),
                        clip_factor=self.config.agc_clip_factor,
                    )
                elif self.config.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )

                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        if self.is_main:
                            print(f"WARNING: NaN/Inf gradient at step {self.global_step}, skipping")
                        self.optimizer.zero_grad(set_to_none=True)
                        continue

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            self.global_step += 1
            self._step_curriculum_dropout()

            if self.global_step % self.config.log_steps == 0:
                if self.world_size > 1:
                    accumulated_loss = all_reduce_mean(accumulated_loss)

                accumulated_loss_val = accumulated_loss.item()

                if self.is_main:
                    elapsed = time.time() - step_start_time
                    steps_per_sec = self.config.log_steps / elapsed
                    samples_per_sec = steps_per_sec * self.config.batch_size * self.world_size
                    lr = self.scheduler.get_last_lr()[0]
                    mem_stats = CUDAMemoryManager.get_memory_stats()

                    print(
                        f"Step {self.global_step}/{self.config.max_steps} | "
                        f"Loss: {accumulated_loss_val:.4f} | "
                        f"LR: {lr:.2e} | "
                        f"Steps/s: {steps_per_sec:.2f} | "
                        f"Samples/s: {samples_per_sec:.1f} | "
                        f"Mem: {mem_stats['allocated']:.1f}/{mem_stats['reserved']:.1f}GB"
                    )

                accumulated_loss = torch.tensor(0.0, device=self.device)
                step_start_time = time.time()

            if self.global_step % self.config.eval_steps == 0 and self.eval_fn:
                self._evaluate()

            if self.global_step % self.config.save_steps == 0:
                self._save_checkpoint()

        if self.is_main:
            print(f"Training complete! Final step: {self.global_step}")

        cleanup_distributed()

    def _training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        assert self.model.training, "Model must be in train mode during _training_step"

        if "input_ids" in batch:
            # MQAR decoder-only mode
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            logits = self.model(None, input_ids)
            logits_flat = logits.reshape(-1, logits.size(-1))
            labels_flat = labels.reshape(-1)
            loss = self.loss_fn(logits_flat, labels_flat)
        else:
            # Seq2seq NMT or MQAR seq2seq mode
            src_ids = batch["src_ids"]
            tgt_ids = batch["tgt_ids"]
            labels = batch.get("labels", tgt_ids[:, 1:])
            decoder_input = tgt_ids[:, :-1]

            # Shape invariant: decoder input and labels must align for teacher forcing
            assert decoder_input.size(1) == labels.size(1), (
                f"Decoder input length {decoder_input.size(1)} != labels length {labels.size(1)}. "
                f"For custom labels (e.g., MQAR), ensure labels have length = tgt_ids.size(1) - 1"
            )

            src_mask = (src_ids != 0).float()
            logits = self.model(src_ids, decoder_input, src_mask=src_mask)
            loss = self.loss_fn(logits, labels)

        return loss

    def _compute_masked_loss(self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits_flat = logits.reshape(-1, logits.size(-1))
        labels_flat = labels.reshape(-1)
        loss = self.loss_fn(logits_flat, labels_flat)
        return loss

    def _step_curriculum_dropout(self):
        """Step all CurriculumDropout modules to update their annealing schedule."""
        model = get_unwrapped_model(self.model)
        for module in model.modules():
            if isinstance(module, CurriculumDropout):
                module.step()

    def _evaluate(self):
        if self.eval_fn is None or self.eval_dataloader is None:
            return

        self.model.eval()
        metrics = self.eval_fn(self.model, self.eval_dataloader, self.device)
        self.model.train()

        if self.is_main:
            print(f"Eval @ step {self.global_step}: {metrics}")
            if "loss" in metrics and metrics["loss"] < self.best_metric:
                self.best_metric = metrics["loss"]
                self._save_checkpoint("best")

        barrier()

    def _save_checkpoint(self, name: Optional[str] = None):
        if not self.is_main:
            barrier()
            return

        if name is None:
            name = f"checkpoint-{self.global_step}"

        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        model_to_save = get_unwrapped_model(self.model)

        config_dict = None
        if hasattr(model_to_save, 'config'):
            config_dict = asdict(model_to_save.config)

        checkpoint = {
            'model_state_dict': model_to_save.state_dict(),
            'config': config_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_metric': self.best_metric,
            # RNG states for reproducible resume
            'rng_state': {
                'python': random.getstate(),
                'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            'metadata': {
                'pytorch_version': torch.__version__,
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                'timestamp': datetime.now().isoformat(),
                'world_size': self.world_size,
            }
        }

        torch.save(checkpoint, checkpoint_dir / "checkpoint.pt")
        print(f"Saved checkpoint to {checkpoint_dir}")

        self._cleanup_checkpoints()
        barrier()

    def _cleanup_checkpoints(self):
        checkpoints = sorted(
            [d for d in self.output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda x: int(x.name.split("-")[1]),
        )

        while len(checkpoints) > self.config.save_total_limit:
            oldest = checkpoints.pop(0)
            shutil.rmtree(oldest)
            print(f"Removed old checkpoint: {oldest}")

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint_dir = Path(checkpoint_path)

        model_to_load = get_unwrapped_model(self.model)

        unified_path = checkpoint_dir / "checkpoint.pt"
        if not unified_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {unified_path}")

        checkpoint = torch.load(unified_path, map_location=self.device, weights_only=False)
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('scheduler_state_dict') is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_metric = checkpoint['best_metric']

        if 'rng_state' in checkpoint:
            rng_state = checkpoint['rng_state']
            random.setstate(rng_state['python'])
            np.random.set_state(rng_state['numpy'])
            torch.set_rng_state(rng_state['torch'])
            if rng_state['cuda'] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng_state['cuda'])
            if self.is_main:
                print("  Restored RNG states")

        if 'metadata' in checkpoint:
            print(f"Loaded checkpoint from {checkpoint_dir}")
            print(f"  Step: {self.global_step}, Epoch: {self.epoch}")
