"""NMT Trainer with H100/BF16 optimization and distributed training support."""

import os
import time
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Callable, Iterator
import math
from dataclasses import dataclass, asdict

import psutil
import numpy as np
import torch
import torch.nn as nn
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
from ..models.utils import get_unwrapped_model, split_params_for_weight_decay
from ..models.align_mamba import CurriculumDropout
from ..constants import (
    ADAM_BETAS, ADAM_EPS, LOG_STEPS, EVAL_STEPS, SAVE_STEPS,
    MIN_LR, MIN_WARMUP_STEPS, WEIGHT_DECAY, WARMUP_RATIO,
    USE_BF16, USE_COMPILE, COMPILE_MODE, USE_AGC, AGC_CLIP_FACTOR,
    GRADIENT_CHECKPOINTING,
)


def adaptive_gradient_clip_(
    parameters: Iterator[nn.Parameter],
    clip_factor: float = AGC_CLIP_FACTOR,
    eps: float = 1e-3,
) -> None:
    """Adaptive Gradient Clipping (AGC) from NFNet paper (Brock et al., 2021)."""
    for p in parameters:
        if p.grad is None:
            continue
        param_norm = p.data.norm(p=2).clamp(min=eps)
        grad_norm = p.grad.data.norm(p=2)
        max_norm = param_norm * clip_factor
        if grad_norm > max_norm:
            p.grad.data.mul_(max_norm / (grad_norm + eps))


def compute_adaptive_smoothing(vocab_size: int, base_smoothing: float = 0.1) -> float:
    """Compute label smoothing scaled by vocabulary size."""
    return base_smoothing * math.log2(vocab_size) / math.log2(32768)


@dataclass
class NMTTrainerConfig:
    """Simplified trainer config - infrastructure values hardcoded in constants.py."""
    seed: int = 42
    max_steps: int = 5000
    batch_size: int = 64
    learning_rate: float = 1e-4
    label_smoothing: Optional[float] = None  # None = adaptive
    output_dir: str = "outputs"


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

        if GRADIENT_CHECKPOINTING and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()

        # torch.compile BEFORE distributed wrapping for optimal performance
        if USE_COMPILE:
            self.model = torch.compile(self.model, mode=COMPILE_MODE)

        if self.world_size > 1:
            dist_config = DistributedConfig(strategy="ddp")
            self.model = wrap_model_distributed(
                self.model, dist_config, self.device, use_bf16=USE_BF16
            )
        else:
            self.model = self.model.to(self.device)

        self.optimizer = self._create_optimizer()

        # Adaptive warmup: ratio of max_steps with minimum floor
        adaptive_warmup = max(
            int(self.config.max_steps * WARMUP_RATIO),
            MIN_WARMUP_STEPS,
        )
        if self.is_main:
            print(f"Warmup steps: {adaptive_warmup} ({WARMUP_RATIO*100:.0f}% of {self.config.max_steps})")

        self.scheduler = create_scheduler(
            optimizer=self.optimizer,
            warmup_steps=adaptive_warmup,
            max_steps=self.config.max_steps,
            min_lr=MIN_LR,
        )

        # Adaptive label smoothing based on vocab size
        if self.config.label_smoothing is not None:
            smoothing = self.config.label_smoothing
        else:
            vocab_size = getattr(getattr(model, 'config', None), 'vocab_size', 8192)
            smoothing = compute_adaptive_smoothing(vocab_size)
            if self.is_main:
                print(f"Adaptive label smoothing: {smoothing:.4f} (vocab_size={vocab_size})")

        self.loss_fn = create_loss_fn(smoothing=smoothing)

        self.global_step = 0
        self.epoch = 0
        self.best_metric = float("inf")

        self.output_dir = Path(self.config.output_dir)
        if self.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        barrier()

    def _setup_h100_optimizations(self):
        """Apply H100 optimizations. Assumes CUDA with Ampere+ GPU."""
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        if self.world_size > 1:
            os.environ.setdefault("NCCL_P2P_LEVEL", "NVL")
            os.environ.setdefault("NCCL_P2P_DISABLE", "0")
            os.environ.setdefault("NCCL_IB_DISABLE", "1")
            os.environ.setdefault("NCCL_DEBUG", "WARN")
            os.environ.setdefault("NCCL_SOCKET_IFNAME", "^lo,docker")

        if self.is_main:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU: {gpu_name} ({gpu_mem:.0f}GB)")
            print(f"Optimizations: TF32=Y, FlashSDP=Y, cuDNN=Y")

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
        optimizer_groups = split_params_for_weight_decay(self.model, WEIGHT_DECAY)
        use_fused = self.device.type == "cuda"
        if self.is_main:
            print(f"Using AdamW optimizer (fused={use_fused})")
        return torch.optim.AdamW(
            optimizer_groups,
            lr=self.config.learning_rate,
            betas=ADAM_BETAS,
            eps=ADAM_EPS,
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
        total_cores = psutil.cpu_count(logical=False) or 8
        available = max(1, total_cores - 4)
        num_workers = min(16, max(1, available // max(1, world_size)))

        sampler = None
        shuffle = is_train
        if world_size > 1:
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
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
        accumulated_loss = torch.tensor(0.0, device=self.device)
        step_start_time = time.time()

        if hasattr(self.train_dataloader, 'sampler') and hasattr(self.train_dataloader.sampler, 'set_epoch'):
            self.train_dataloader.sampler.set_epoch(self.epoch)

        while self.global_step < self.config.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                self.epoch += 1
                if hasattr(self.train_dataloader, 'sampler') and hasattr(self.train_dataloader.sampler, 'set_epoch'):
                    self.train_dataloader.sampler.set_epoch(self.epoch)
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)

            batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            with autocast(dtype=torch.bfloat16, enabled=USE_BF16):
                loss = self._training_step(batch)

            if torch.isnan(loss) or torch.isinf(loss):
                if self.is_main:
                    print(f"WARNING: NaN/Inf loss at step {self.global_step}, skipping")
                self.optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()
            accumulated_loss = accumulated_loss + loss.detach()

            # Adaptive Gradient Clipping
            if USE_AGC:
                adaptive_gradient_clip_(self.model.parameters())

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

            self.global_step += 1
            self._step_curriculum_dropout()

            if self.global_step % LOG_STEPS == 0:
                if self.world_size > 1:
                    accumulated_loss = all_reduce_mean(accumulated_loss)

                if self.is_main:
                    elapsed = time.time() - step_start_time
                    steps_per_sec = LOG_STEPS / elapsed
                    samples_per_sec = steps_per_sec * self.config.batch_size * self.world_size
                    lr = self.scheduler.get_last_lr()[0]
                    mem_alloc = torch.cuda.memory_allocated() / (1024**3)
                    mem_reserved = torch.cuda.memory_reserved() / (1024**3)

                    print(
                        f"Step {self.global_step}/{self.config.max_steps} | "
                        f"Loss: {accumulated_loss.item():.4f} | "
                        f"LR: {lr:.2e} | "
                        f"Steps/s: {steps_per_sec:.2f} | "
                        f"Samples/s: {samples_per_sec:.1f} | "
                        f"Mem: {mem_alloc:.1f}/{mem_reserved:.1f}GB"
                    )

                accumulated_loss = torch.tensor(0.0, device=self.device)
                step_start_time = time.time()

            if self.global_step % EVAL_STEPS == 0 and self.eval_fn:
                self._evaluate()

            if self.global_step % SAVE_STEPS == 0:
                self._save_checkpoint()

        if self.is_main:
            print(f"Training complete! Final step: {self.global_step}")
        cleanup_distributed()

    def _training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        assert self.model.training, "Model must be in train mode"

        labels = batch.get("labels")
        if labels is not None:
            valid_labels = labels[labels != -100]
            if valid_labels.numel() > 0:
                model = get_unwrapped_model(self.model)
                vocab_size = getattr(model.config, 'vocab_size', 8192)
                assert valid_labels.min() >= 0, f"Negative label: {valid_labels.min().item()}"
                assert valid_labels.max() < vocab_size, f"Label {valid_labels.max().item()} exceeds vocab {vocab_size}"

        if "input_ids" in batch:
            # MQAR decoder-only mode
            logits = self.model(None, batch["input_ids"])
            loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), batch["labels"].reshape(-1))
        else:
            # Seq2seq mode
            src_ids, tgt_ids = batch["src_ids"], batch["tgt_ids"]
            labels = batch.get("labels", tgt_ids[:, 1:])
            decoder_input = tgt_ids[:, :-1]
            src_mask = (src_ids != 0).float()
            logits = self.model(src_ids, decoder_input, src_mask=src_mask)
            loss = self.loss_fn(logits, labels)

        return loss

    def _step_curriculum_dropout(self):
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

        name = name or f"checkpoint-{self.global_step}"
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        model_to_save = get_unwrapped_model(self.model)
        config_dict = asdict(model_to_save.config) if hasattr(model_to_save, 'config') else None

        checkpoint = {
            'model_state_dict': model_to_save.state_dict(),
            'config': config_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_metric': self.best_metric,
            'rng_state': {
                'python': random.getstate(),
                'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all(),
            },
            'metadata': {
                'pytorch_version': torch.__version__,
                'cuda_version': torch.version.cuda,
                'timestamp': datetime.now().isoformat(),
                'world_size': self.world_size,
            }
        }

        torch.save(checkpoint, checkpoint_dir / "checkpoint.pt")
        print(f"Saved checkpoint to {checkpoint_dir}")
        self._cleanup_checkpoints()
        barrier()

    def _cleanup_checkpoints(self, keep_limit: int = 3):
        checkpoints = sorted(
            [d for d in self.output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda x: int(x.name.split("-")[1]),
        )
        while len(checkpoints) > keep_limit:
            oldest = checkpoints.pop(0)
            shutil.rmtree(oldest)
            print(f"Removed old checkpoint: {oldest}")

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint_dir = Path(checkpoint_path)
        unified_path = checkpoint_dir / "checkpoint.pt"
        if not unified_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {unified_path}")

        model_to_load = get_unwrapped_model(self.model)
        checkpoint = torch.load(unified_path, map_location=self.device, weights_only=False)
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_metric = checkpoint['best_metric']

        if 'rng_state' in checkpoint:
            rng_state = checkpoint['rng_state']
            random.setstate(rng_state['python'])
            np.random.set_state(rng_state['numpy'])
            torch.set_rng_state(rng_state['torch'])
            if rng_state.get('cuda'):
                torch.cuda.set_rng_state_all(rng_state['cuda'])

        if self.is_main:
            print(f"Loaded checkpoint: step={self.global_step}, epoch={self.epoch}")
