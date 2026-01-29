"""Distributed training utilities for multi-GPU training (DDP/FSDP)."""

import os
import socket
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from functools import partial


class DistributedStrategy(Enum):
    NONE = "none"
    DDP = "ddp"
    FSDP = "fsdp"
    FSDP_FULL = "fsdp_full"


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    strategy: str = "ddp"
    gradient_as_bucket_view: bool = True
    static_graph: bool = True  # Required for torch.compile compatibility
    # 512MB bucket optimal for NVLink - reduces NCCL sync overhead vs default 25MB
    bucket_cap_mb: int = 512
    sharding_strategy: str = "full_shard"
    cpu_offload: bool = False
    backward_prefetch: str = "backward_pre"
    min_num_params: int = 1_000_000
    fsdp_mixed_precision: bool = True
    backend: str = "nccl"
    master_addr: str = "localhost"
    master_port: str = "29500"


def setup_distributed(config: Optional[DistributedConfig] = None) -> Dict[str, Any]:
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
        # Multi-GPU detected but not using torchrun
        # Recommend torchrun for proper multi-GPU setup
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
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
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


def get_fsdp_mixed_precision(use_bf16: bool = True) -> MixedPrecision:
    """Get FSDP mixed precision policy."""
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    return MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)


def get_fsdp_sharding_strategy(strategy: str) -> ShardingStrategy:
    """Get FSDP sharding strategy."""
    strategies = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
        "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
    }
    return strategies.get(strategy, ShardingStrategy.FULL_SHARD)


def get_backward_prefetch(prefetch: Optional[str]) -> Optional[BackwardPrefetch]:
    """Get FSDP backward prefetch policy."""
    if prefetch is None:
        return None
    policies = {
        "backward_pre": BackwardPrefetch.BACKWARD_PRE,
        "backward_post": BackwardPrefetch.BACKWARD_POST,
    }
    return policies.get(prefetch, BackwardPrefetch.BACKWARD_PRE)


def wrap_model_distributed(
    model: nn.Module,
    config: DistributedConfig,
    device: torch.device,
    use_bf16: bool = True,
) -> nn.Module:
    """Wrap model for distributed training (DDP or FSDP)."""
    strategy = DistributedStrategy(config.strategy)

    if strategy == DistributedStrategy.NONE:
        return model.to(device)

    if not dist.is_initialized():
        print("Warning: Distributed not initialized, using single GPU")
        return model.to(device)

    if strategy == DistributedStrategy.DDP:
        model = model.to(device)
        # Larger bucket_cap_mb reduces NCCL calls - optimal for high-bandwidth NVLink
        model = DDP(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            gradient_as_bucket_view=config.gradient_as_bucket_view,
            static_graph=config.static_graph,
            bucket_cap_mb=config.bucket_cap_mb,
        )

    elif strategy in (DistributedStrategy.FSDP, DistributedStrategy.FSDP_FULL):
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=config.min_num_params,
        )

        mixed_precision = None
        if config.fsdp_mixed_precision:
            mixed_precision = get_fsdp_mixed_precision(use_bf16)

        sharding_strategy = get_fsdp_sharding_strategy(config.sharding_strategy)
        backward_prefetch = get_backward_prefetch(config.backward_prefetch)
        cpu_offload = CPUOffload(offload_params=True) if config.cpu_offload else None

        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision,
            sharding_strategy=sharding_strategy,
            backward_prefetch=backward_prefetch,
            cpu_offload=cpu_offload,
            device_id=device,
            use_orig_params=True,  # Required for torch.compile
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
