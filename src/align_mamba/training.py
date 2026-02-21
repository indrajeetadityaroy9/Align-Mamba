# Training loop for Align-Mamba.
import json
import math
import shutil
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from accelerate import Accelerator

from align_mamba.config import Config
from align_mamba.model import HybridMambaEncoderDecoder
from align_mamba.kernels.loss import fused_cross_entropy_loss

_WARMUP_RATIO = 0.05
_LR_FLOOR = 0.01
_ADAM_BETAS = (0.9, 0.999)
_ADAM_EPS = 1e-8
_LOG_EPOCH_INTERVAL = 1
_EVAL_EPOCH_INTERVAL = 1
_SAVE_EPOCH_INTERVAL = 1
_MAX_CHECKPOINTS = 3
_GRAD_CLIP_FACTOR = 2.0
_GRAD_EMA_DECAY = 0.98


def _emit_log(**payload) -> None:
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True))


class CosineScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, *, max_steps: int):
        self.optimizer = optimizer
        self.warmup = int(_WARMUP_RATIO * max_steps)
        self.max_steps = max_steps
        self.base_lrs = [g['lr'] for g in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count < self.warmup:
            scale = self.step_count / self.warmup
        else:
            progress = (self.step_count - self.warmup) / (self.max_steps - self.warmup)
            scale = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
        scale = max(scale, _LR_FLOOR)
        for g, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            g['lr'] = base_lr * scale

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


class Trainer:
    def __init__(
        self,
        model: HybridMambaEncoderDecoder,
        train_loader: DataLoader,
        config: Config,
        accelerator: Accelerator,
        eval_loader: DataLoader,
    ):
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

        self.config = config
        self.accelerator = accelerator
        self.device = accelerator.device
        self.is_main = accelerator.is_main_process
        self.world_size = accelerator.num_processes
        self.output_dir = Path(config.output_dir)

        model = torch.compile(model, mode="reduce-overhead")

        no_decay = ["bias", "norm"]
        params_decay = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
        params_no_decay = [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
        groups = [
            {"params": params_decay, "weight_decay": config.weight_decay},
            {"params": params_no_decay, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(groups, lr=config.learning_rate,
                                       betas=_ADAM_BETAS, eps=_ADAM_EPS, fused=True)

        self.model, self.optimizer, self.train_loader, self.eval_loader = (
            accelerator.prepare(model, optimizer, train_loader, eval_loader)
        )

        self.scheduler = CosineScheduler(self.optimizer, max_steps=config.max_steps)

        self.steps_per_epoch = max(1, config.num_samples // config.batch_size)
        self.log_steps = self.steps_per_epoch * _LOG_EPOCH_INTERVAL
        self.eval_steps = self.steps_per_epoch * _EVAL_EPOCH_INTERVAL
        self.save_steps = self.steps_per_epoch * _SAVE_EPOCH_INTERVAL

        self.global_step = 0
        self.epoch = 0
        self.best_acc = 0.0
        self._grad_norm_ema = None

        if self.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def resume_from(self, path: str) -> None:
        # Resume training state from a checkpoint directory.
        ckpt = torch.load(Path(path) / "checkpoint.pt", map_location="cpu", weights_only=True)
        self.accelerator.unwrap_model(self.model).load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.global_step = ckpt["global_step"]
        self.epoch = ckpt["epoch"]
        self.best_acc = ckpt["best_acc"]
        self.scheduler.step_count = ckpt["scheduler_step"]
        if self.is_main:
            _emit_log(
                event="train_resume",
                checkpoint=str(path),
                step=self.global_step,
                best_acc=round(self.best_acc, 6),
            )

    def train(self):
        self.model.train()
        data_iter = iter(self.train_loader)
        loss_accum = torch.tensor(0.0, device=self.device)
        start_time = time.time()

        while self.global_step < self.config.max_steps:
            batch = next(data_iter, None)
            if batch is None:
                self.epoch += 1
                self.train_loader.set_epoch(self.epoch)
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

            with self.accelerator.autocast():
                logits = self.model(batch["src_ids"], batch["tgt_ids"][:, :-1])
                loss = fused_cross_entropy_loss(logits, batch["labels"], smoothing=self.config.label_smoothing)

            self.accelerator.backward(loss)
            loss_accum += loss.detach()

            if self._grad_norm_ema is None:
                grad_norm = self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )
                self._grad_norm_ema = grad_norm.item()
            else:
                clip_val = self._grad_norm_ema * _GRAD_CLIP_FACTOR
                grad_norm = self.accelerator.clip_grad_norm_(
                    self.model.parameters(), clip_val
                )
                self._grad_norm_ema = (
                    _GRAD_EMA_DECAY * self._grad_norm_ema
                    + (1 - _GRAD_EMA_DECAY) * grad_norm.item()
                )

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1

            if self.global_step % self.log_steps == 0:
                avg_loss = self.accelerator.gather(loss_accum).mean().item() / self.log_steps
                elapsed = time.time() - start_time
                throughput = self.log_steps * self.config.batch_size * self.world_size / elapsed

                if self.is_main:
                    payload = {
                        "event": "train_epoch",
                        "epoch": self.epoch + 1,
                        "step": self.global_step,
                        "max_steps": self.config.max_steps,
                        "loss": round(avg_loss, 6),
                        "lr": self.scheduler.get_lr(),
                        "samples_per_s": round(throughput, 2),
                    }
                    if torch.cuda.is_available():
                        payload["gpu_mem_gb"] = round(torch.cuda.memory_allocated(self.device) / 1e9, 3)
                    _emit_log(**payload)

                loss_accum = torch.tensor(0.0, device=self.device)
                start_time = time.time()

            if self.global_step % self.eval_steps == 0:
                self._eval()

            if self.global_step % self.save_steps == 0:
                self._save()

        if self.is_main:
            _emit_log(
                event="train_complete",
                step=self.global_step,
                max_steps=self.config.max_steps,
                best_acc=round(self.best_acc, 6),
            )

    def _eval(self):
        self.model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for batch in self.eval_loader:
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                logits = self.model(batch["src_ids"], batch["tgt_ids"][:, :-1])
                preds = logits.argmax(dim=-1)
                labels = batch["labels"]
                mask = labels != -100
                correct += ((preds == labels) & mask).sum().item()
                total += mask.sum().item()

        counts = torch.tensor([correct, total], device=self.device, dtype=torch.float64)
        counts = self.accelerator.reduce(counts, reduction="sum")
        acc = (counts[0] / counts[1]).item()
        if self.is_main:
            _emit_log(
                event="eval",
                step=self.global_step,
                epoch=self.epoch + 1,
                token_accuracy=round(acc, 6),
            )

        if acc > self.best_acc:
            self.best_acc = acc
            self._save("best")

        self.model.train()

    def _save(self, name: str = ""):
        if not self.is_main:
            return

        if not name:
            name = f"checkpoint-{self.global_step}"
        path = self.output_dir / name
        path.mkdir(parents=True, exist_ok=True)

        unwrapped = self.accelerator.unwrap_model(self.model)
        self.accelerator.save({
            'model_state_dict': self.accelerator.get_state_dict(self.model),
            'config': asdict(unwrapped.config),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_step': self.scheduler.step_count,
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_acc': self.best_acc,
            'timestamp': datetime.now().isoformat(),
        }, path / "checkpoint.pt")

        _emit_log(
            event="checkpoint_saved",
            step=self.global_step,
            name=name,
            path=str(path),
            best_acc=round(self.best_acc, 6),
        )

        ckpts = sorted([d for d in self.output_dir.iterdir() if d.name.startswith("checkpoint-")],
                      key=lambda x: int(x.name.split("-")[1]))
        while len(ckpts) > _MAX_CHECKPOINTS:
            shutil.rmtree(ckpts.pop(0))
