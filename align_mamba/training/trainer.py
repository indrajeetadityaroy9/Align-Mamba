"""NMT Trainer with distributed training support."""

import time
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .optimization import (
    LabelSmoothingCrossEntropy,
    CosineAnnealingWarmupScheduler,
    compute_logging_intervals,
    create_adaptive_param_groups,
    adaptive_gradient_clip_,
    compute_label_smoothing_from_entropy,
)
from .distributed import setup_distributed, cleanup_distributed, barrier
from align_mamba.models.encoder_decoder import get_unwrapped_model
from align_mamba.config import (
    ADAM_BETAS, ADAM_EPS, MIN_WARMUP_STEPS,
    USE_BF16, USE_COMPILE, COMPILE_MODE, GRADIENT_CHECKPOINTING,
)
from align_mamba.evaluation.metrics import compute_batch_metrics, BatchMetrics

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.set_float32_matmul_precision("high")
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)


class NMTTrainer:
    """Training loop with adaptive hyperparameters."""

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        seed: int = 42,
        max_steps: int = 5000,
        batch_size: int = 64,
        learning_rate: float = 1e-4,
        gradient_accumulation_steps: int = 1,
        label_smoothing: Optional[float] = None,
        output_dir: str = "outputs",
        eval_dataloader: Optional[DataLoader] = None,
        dist_info: Optional[Dict[str, Any]] = None,
        num_samples: Optional[int] = None,
    ):
        self.seed = seed
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.label_smoothing = label_smoothing
        self.output_dir = Path(output_dir)

        num_samples = num_samples or len(train_dataloader.dataset)
        intervals = compute_logging_intervals(num_samples, batch_size)
        self.log_steps = intervals["log_steps"]
        self.eval_steps = intervals["eval_steps"]
        self.save_steps = intervals["save_steps"]

        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        self.dist_info = dist_info if dist_info is not None else setup_distributed()
        self.device = self.dist_info["device"]
        self.is_main = self.dist_info["is_main"]
        self.world_size = self.dist_info["world_size"]

        if self.is_main:
            print(f"world_size={self.world_size} device={self.device}")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if GRADIENT_CHECKPOINTING:
            self.model.gradient_checkpointing_enable()

        if USE_COMPILE:
            self.model = torch.compile(self.model, mode=COMPILE_MODE)

        self.model = self.model.to(self.device)
        if self.world_size > 1 and dist.is_initialized():
            self.model = DDP(
                self.model,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
                gradient_as_bucket_view=True,
                find_unused_parameters=False,
            )

        self.optimizer = self._create_optimizer()

        self.scheduler = CosineAnnealingWarmupScheduler(
            self.optimizer,
            warmup_steps=MIN_WARMUP_STEPS,
            max_steps=self.max_steps,
            min_lr=0.0,
        )

        if self.label_smoothing is not None:
            smoothing = self.label_smoothing
        else:
            vocab_size = getattr(getattr(model, 'config', None), 'vocab_size', 8192)
            smoothing = compute_label_smoothing_from_entropy(vocab_size)

        self.loss_fn = LabelSmoothingCrossEntropy(smoothing=smoothing)

        self.global_step = 0
        self.epoch = 0
        self.best_metric = float("inf")

        if self.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        barrier()

    def _create_optimizer(self) -> torch.optim.Optimizer:
        optimizer_groups = create_adaptive_param_groups(
            self.model, self.learning_rate, base_decay=0.01
        )
        return torch.optim.AdamW(
            optimizer_groups,
            betas=ADAM_BETAS,
            eps=ADAM_EPS,
            fused=True,
        )

    def train(self):
        self.model.train()
        data_iter = iter(self.train_dataloader)
        accumulated_loss = torch.tensor(0.0, device=self.device)
        step_start_time = time.time()
        micro_step = 0

        if hasattr(self.train_dataloader, 'sampler') and hasattr(self.train_dataloader.sampler, 'set_epoch'):
            self.train_dataloader.sampler.set_epoch(self.epoch)

        while self.global_step < self.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                self.epoch += 1
                if hasattr(self.train_dataloader, 'sampler') and hasattr(self.train_dataloader.sampler, 'set_epoch'):
                    self.train_dataloader.sampler.set_epoch(self.epoch)
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)

            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=USE_BF16):
                loss = self._training_step(batch)

            if torch.isnan(loss) or torch.isinf(loss):
                if self.is_main:
                    print(f"NaN/Inf loss at step {self.global_step}, skipping")
                self.optimizer.zero_grad(set_to_none=True)
                continue

            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            accumulated_loss = accumulated_loss + loss.detach()
            micro_step += 1

            if micro_step % self.gradient_accumulation_steps == 0:
                adaptive_gradient_clip_(self.model.parameters())
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

                if self.global_step % self.log_steps == 0:
                    if self.world_size > 1 and dist.is_initialized():
                        dist.all_reduce(accumulated_loss, op=dist.ReduceOp.AVG)

                    elapsed = time.time() - step_start_time
                    steps_per_sec = self.log_steps / elapsed
                    samples_per_sec = steps_per_sec * self.batch_size * self.world_size * self.gradient_accumulation_steps
                    lr = self.scheduler.get_last_lr()[0]
                    mem_alloc = torch.cuda.memory_allocated() / (1024**3)

                    if self.is_main:
                        print(
                            f"step={self.global_step}/{self.max_steps} loss={accumulated_loss.item():.4f} "
                            f"lr={lr:.2e} throughput={samples_per_sec:.0f}samples/s mem={mem_alloc:.1f}GB"
                        )

                    accumulated_loss = torch.tensor(0.0, device=self.device)
                    step_start_time = time.time()

                if self.global_step % self.eval_steps == 0 and self.eval_dataloader is not None:
                    self._evaluate()

                if self.global_step % self.save_steps == 0:
                    self._save_checkpoint()

        if self.is_main:
            print(f"training_complete step={self.global_step}")
        cleanup_distributed()

    def _training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "input_ids" in batch:
            logits = self.model(None, batch["input_ids"])
            return self.loss_fn(logits.reshape(-1, logits.size(-1)), batch["labels"].reshape(-1))
        else:
            src_ids, tgt_ids = batch["src_ids"], batch["tgt_ids"]
            labels = batch["labels"]
            decoder_input = tgt_ids[:, :-1]
            src_mask = (src_ids != 0).float()
            logits = self.model(src_ids, decoder_input, src_mask=src_mask)
            return self.loss_fn(logits, labels)

    def _evaluate(self):
        self.model.eval()
        accumulated = BatchMetrics(0, 0, 0, 0)

        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

                if "input_ids" in batch:
                    logits = self.model(None, batch["input_ids"])
                else:
                    src_ids, tgt_ids = batch["src_ids"], batch["tgt_ids"]
                    decoder_input = tgt_ids[:, :-1]
                    src_mask = (src_ids != 0).float()
                    logits = self.model(src_ids, decoder_input, src_mask=src_mask)

                predictions = logits.argmax(dim=-1)
                labels = batch["labels"]
                mask = labels != -100

                batch_metrics = compute_batch_metrics(predictions, labels, mask)
                accumulated = accumulated + batch_metrics

        if self.is_main:
            print(f"eval step={self.global_step} token_acc={accumulated.token_accuracy:.4f} sample_acc={accumulated.sample_accuracy:.4f}")

        if accumulated.sample_accuracy > (1.0 - self.best_metric):
            self.best_metric = 1.0 - accumulated.sample_accuracy
            self._save_checkpoint("best")

        self.model.train()
        barrier()

    def _save_checkpoint(self, name: Optional[str] = None):
        if not self.is_main:
            barrier()
            return

        name = name or f"checkpoint-{self.global_step}"
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        model_to_save = get_unwrapped_model(self.model)
        config_dict = asdict(model_to_save.config)

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
        print(f"checkpoint_saved path={checkpoint_dir}")
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

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint_dir = Path(checkpoint_path)
        unified_path = checkpoint_dir / "checkpoint.pt"

        model_to_load = get_unwrapped_model(self.model)
        checkpoint = torch.load(unified_path, map_location=self.device, weights_only=False)
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_metric = checkpoint['best_metric']

        rng_state = checkpoint['rng_state']
        random.setstate(rng_state['python'])
        np.random.set_state(rng_state['numpy'])
        torch.set_rng_state(rng_state['torch'])
        torch.cuda.set_rng_state_all(rng_state['cuda'])

        if self.is_main:
            print(f"checkpoint_loaded step={self.global_step} epoch={self.epoch}")
