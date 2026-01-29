"""Distributed training utilities for multi-GPU DDP training."""

import os
from typing import Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass
class DistributedConfig:
    """DDP configuration for multi-GPU training."""
    strategy: str = "ddp"
    gradient_as_bucket_view: bool = True
    static_graph: bool = True  # Required for torch.compile compatibility
    # 512MB bucket optimal for NVLink - reduces NCCL sync overhead vs default 25MB
    bucket_cap_mb: int = 512
    backend: str = "nccl"
    master_addr: str = "localhost"
    master_port: str = "29500"


def setup_distributed(config: DistributedConfig = None) -> Dict[str, Any]:
    """Setup distributed training environment. Returns dict with rank, local_rank, world_size, device."""
    config = config or DistributedConfig()

    if dist.is_initialized():
        return {
            "rank": dist.get_rank(),
            "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
            "world_size": dist.get_world_size(),
            "device": torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"),
            "is_main": dist.get_rank() == 0,
        }

    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    elif torch.cuda.device_count() > 1:
        gpu_count = torch.cuda.device_count()
        print(f"\n{'='*60}")
        print(f"MULTI-GPU SETUP ({gpu_count} GPUs detected)")
        print(f"{'='*60}")
        print(f"For multi-GPU training, use torchrun:")
        print(f"  torchrun --nproc_per_node={gpu_count} <script.py> ...")
        print(f"{'='*60}\n")
        rank = 0
        local_rank = 0
        world_size = 1
    else:
        return {
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "device": torch.device("cuda"),
            "is_main": True,
        }

    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", config.master_addr)
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", config.master_port)

    if not dist.is_initialized() and world_size > 1:
        dist.init_process_group(
            backend=config.backend,
            rank=rank,
            world_size=world_size,
        )

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    return {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": device,
        "is_main": rank == 0,
    }


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_model_distributed(
    model: nn.Module,
    config: DistributedConfig,
    device: torch.device,
    use_bf16: bool = True,
) -> nn.Module:
    """Wrap model for DDP distributed training."""
    if config.strategy == "none":
        return model.to(device)

    if not dist.is_initialized():
        print("Warning: Distributed not initialized, using single GPU")
        return model.to(device)

    model = model.to(device)
    model = DDP(
        model,
        device_ids=[device.index] if device.type == "cuda" else None,
        gradient_as_bucket_view=config.gradient_as_bucket_view,
        static_graph=config.static_graph,
        bucket_cap_mb=config.bucket_cap_mb,
    )

    return model


def get_nvlink_info() -> Dict[str, Any]:
    """Get NVLink topology information."""
    info = {
        "gpu_count": torch.cuda.device_count(),
        "nvlink_available": False,
        "p2p_available": [],
    }

    if info["gpu_count"] < 2:
        return info

    for i in range(info["gpu_count"]):
        for j in range(info["gpu_count"]):
            if i != j:
                can_access = torch.cuda.can_device_access_peer(i, j)
                info["p2p_available"].append((i, j, can_access))
                if can_access:
                    info["nvlink_available"] = True

    return info


def print_distributed_info(dist_info: Dict[str, Any]):
    """Print distributed training information (main rank only)."""
    if not dist_info.get("is_main", True):
        return

    print("\n" + "=" * 60)
    print("DISTRIBUTED TRAINING INFO")
    print("=" * 60)
    print(f"World Size: {dist_info['world_size']}")
    print(f"Rank: {dist_info['rank']}")
    print(f"Local Rank: {dist_info['local_rank']}")
    print(f"Device: {dist_info['device']}")

    nvlink_info = get_nvlink_info()
    print(f"\nGPU Count: {nvlink_info['gpu_count']}")
    print(f"NVLink Available: {nvlink_info['nvlink_available']}")

    if nvlink_info['p2p_available']:
        print("P2P Access:")
        for i, j, can_access in nvlink_info['p2p_available']:
            status = "Y" if can_access else "N"
            print(f"  GPU {i} -> GPU {j}: {status}")

    print("=" * 60 + "\n")


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce tensor and compute mean across processes."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return tensor

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """Broadcast object from source rank to all ranks."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return obj

    object_list = [obj]
    dist.broadcast_object_list(object_list, src=src)
    return object_list[0]


def barrier():
    """Synchronization barrier."""
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier()
