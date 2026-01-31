"""Distributed training utilities."""

import os
from datetime import timedelta
from typing import Any, Dict

import torch
import torch.distributed as dist


def setup_distributed() -> Dict[str, Any]:
    """Setup distributed training. Assumes torchrun launcher for multi-GPU."""
    if dist.is_initialized():
        rank = dist.get_rank()
        info = {
            "rank": rank,
            "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
            "world_size": dist.get_world_size(),
            "device": torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"),
            "is_main": rank == 0,
        }
        return info

    if "RANK" not in os.environ:
        # Single-GPU mode (no distributed launcher)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for training. No CUDA devices found.")
        info = {
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "device": torch.device("cuda:0"),
            "is_main": True,
        }
        return info

    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    rank = dist.get_rank()
    info = {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": dist.get_world_size(),
        "device": device,
        "is_main": rank == 0,
    }
    return info


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def barrier():
    """Synchronization barrier."""
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier()
