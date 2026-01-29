"""Hardware detection and H100 optimizations."""

import os
import subprocess
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from contextlib import contextmanager

import torch
import torch.distributed as dist
import psutil
import flash_attn
import mamba_ssm
import triton


@dataclass
class GPUInfo:
    """Information about a single GPU."""
    index: int
    name: str
    compute_capability: Tuple[int, int]
    total_memory_gb: float
    free_memory_gb: float
    is_h100: bool = False
    supports_bf16: bool = False
    supports_tf32: bool = False
    supports_flash_attn: bool = False


@dataclass
class HardwareInfo:
    """Complete hardware information."""
    gpu_count: int = 0
    gpus: List[GPUInfo] = field(default_factory=list)
    nvlink_available: bool = False
    nvlink_pairs: List[Tuple[int, int]] = field(default_factory=list)
    cpu_count: int = 0
    cpu_model: str = ""
    system_memory_gb: float = 0.0
    cuda_version: str = ""
    cudnn_version: str = ""
    torch_version: str = ""
    torch_compile_available: bool = False


def get_gpu_info(index: int) -> GPUInfo:
    """Get detailed information about a specific GPU."""
    props = torch.cuda.get_device_properties(index)

    total_memory = props.total_memory / (1024**3)
    free_memory = (props.total_memory - torch.cuda.memory_allocated(index)) / (1024**3)
    compute_cap = (props.major, props.minor)

    # H100 = compute capability 9.0
    is_h100 = compute_cap >= (9, 0) or "H100" in props.name
    # BF16/TF32/FlashAttn require Ampere+ (8.0)
    supports_ampere = compute_cap >= (8, 0)

    return GPUInfo(
        index=index,
        name=props.name,
        compute_capability=compute_cap,
        total_memory_gb=total_memory,
        free_memory_gb=free_memory,
        is_h100=is_h100,
        supports_bf16=supports_ampere,
        supports_tf32=supports_ampere,
        supports_flash_attn=supports_ampere,
    )


def detect_nvlink() -> Tuple[bool, List[Tuple[int, int]]]:
    """Detect NVLink connections between GPUs."""
    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        return False, []

    nvlink_pairs = []
    for i in range(gpu_count):
        for j in range(gpu_count):
            if i != j and torch.cuda.can_device_access_peer(i, j):
                nvlink_pairs.append((i, j))

    return len(nvlink_pairs) > 0, nvlink_pairs


def detect_hardware() -> HardwareInfo:
    """Detect all available hardware and capabilities."""
    info = HardwareInfo()

    info.torch_version = torch.__version__
    info.torch_compile_available = hasattr(torch, 'compile')

    if torch.cuda.is_available():
        info.cuda_version = torch.version.cuda or ""
        info.cudnn_version = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else ""

    info.gpu_count = torch.cuda.device_count()
    for i in range(info.gpu_count):
        info.gpus.append(get_gpu_info(i))

    info.nvlink_available, info.nvlink_pairs = detect_nvlink()

    info.cpu_count = os.cpu_count() or 1
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if 'model name' in line:
                    info.cpu_model = line.split(':')[1].strip()
                    break
    except (FileNotFoundError, PermissionError):
        info.cpu_model = "Unknown"

    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if 'MemTotal' in line:
                    mem_kb = int(line.split()[1])
                    info.system_memory_gb = mem_kb / (1024**2)
                    break
    except (FileNotFoundError, PermissionError):
        info.system_memory_gb = 0.0

    return info


def print_hardware_info(info: Optional[HardwareInfo] = None):
    """Print formatted hardware information."""
    if info is None:
        info = detect_hardware()

    print("\n" + "=" * 70)
    print("HARDWARE CONFIGURATION")
    print("=" * 70)

    print(f"\nPyTorch: {info.torch_version}")
    print(f"CUDA: {info.cuda_version}")
    print(f"cuDNN: {info.cudnn_version}")
    print(f"torch.compile: {'Y' if info.torch_compile_available else 'N'}")

    print(f"\n{'-' * 70}")
    print(f"GPUs: {info.gpu_count}")
    for gpu in info.gpus:
        h100_tag = " [H100]" if gpu.is_h100 else ""
        print(f"  [{gpu.index}] {gpu.name}{h100_tag}")
        print(f"      Memory: {gpu.free_memory_gb:.1f} / {gpu.total_memory_gb:.1f} GB")
        print(f"      Compute: {gpu.compute_capability[0]}.{gpu.compute_capability[1]}")
        print(f"      BF16: {'Y' if gpu.supports_bf16 else 'N'} | "
              f"TF32: {'Y' if gpu.supports_tf32 else 'N'} | "
              f"FlashAttn: {'Y' if gpu.supports_flash_attn else 'N'}")

    if info.gpu_count > 1:
        print(f"\n{'-' * 70}")
        print(f"NVLink: {'Y' if info.nvlink_available else 'N'}")
        if info.nvlink_pairs:
            pairs_str = ", ".join([f"{i}<->{j}" for i, j in info.nvlink_pairs if i < j])
            print(f"  Pairs: {pairs_str}")

    print(f"\n{'-' * 70}")
    print(f"CPU: {info.cpu_model}")
    print(f"CPU Cores: {info.cpu_count}")
    print(f"System Memory: {info.system_memory_gb:.1f} GB")

    print(f"\n{'-' * 70}")
    print("Packages:")
    print(f"  flash-attn: v{flash_attn.__version__}")
    print(f"  mamba-ssm: v{mamba_ssm.__version__}")
    print(f"  triton: v{triton.__version__}")

    print("=" * 70 + "\n")


def setup_h100_optimizations() -> Dict[str, bool]:
    """Apply H100/Ampere optimizations. Returns dict of applied settings."""
    applied = {}

    # TF32 for ~3x matmul speedup on Tensor Cores
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    applied["tf32"] = True

    # 'high' = BF16 accumulation, best for H100
    torch.set_float32_matmul_precision("high")
    applied["matmul_precision"] = "high"

    # cuDNN autotuning for optimal conv algorithms
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    applied["cudnn_benchmark"] = True

    # Flash SDP (memory-efficient attention)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    applied["flash_sdp"] = True
    applied["mem_efficient_sdp"] = True

    # expandable_segments reduces memory fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    applied["cuda_alloc_conf"] = "expandable_segments:True"

    applied["flash_attn_2"] = True

    return applied


def get_optimal_compiler_backend() -> str:
    """Get optimal torch.compile backend. Inductor is best for H100/Ampere."""
    return "inductor"


def get_optimal_compile_options() -> Dict[str, Any]:
    """Get optimal torch.compile options for H100."""
    options = {
        "mode": "max-autotune",
        "fullgraph": False,
    }

    if is_hopper():
        options["options"] = {
            "triton.cudagraphs": True,  # Reduces kernel launch overhead
            "epilogue_fusion": True,
            "max_autotune": True,
        }

    return options


def setup_nccl_optimizations(
    nvlink_available: bool = True,
    use_infiniband: bool = False,
) -> Dict[str, str]:
    """Setup NCCL optimizations for multi-GPU training."""
    env_vars = {}

    if nvlink_available:
        # NVL = use NVLink for P2P transfers
        os.environ["NCCL_P2P_LEVEL"] = "NVL"
        env_vars["NCCL_P2P_LEVEL"] = "NVL"
        os.environ["NCCL_P2P_DISABLE"] = "0"
        env_vars["NCCL_P2P_DISABLE"] = "0"

    if use_infiniband:
        os.environ["NCCL_IB_DISABLE"] = "0"
        env_vars["NCCL_IB_DISABLE"] = "0"
    else:
        os.environ["NCCL_IB_DISABLE"] = "1"
        env_vars["NCCL_IB_DISABLE"] = "1"

    os.environ.setdefault("NCCL_DEBUG", "WARN")
    env_vars["NCCL_DEBUG"] = os.environ["NCCL_DEBUG"]

    os.environ.setdefault("NCCL_SOCKET_IFNAME", "^lo,docker")
    env_vars["NCCL_SOCKET_IFNAME"] = os.environ["NCCL_SOCKET_IFNAME"]

    return env_vars


class CUDAMemoryManager:
    """Utilities for managing CUDA memory."""

    @staticmethod
    def get_memory_stats(device: Optional[int] = None) -> Dict[str, float]:
        """Get memory statistics in GB."""
        if device is None:
            device = torch.cuda.current_device()

        return {
            "allocated": torch.cuda.memory_allocated(device) / (1024**3),
            "reserved": torch.cuda.memory_reserved(device) / (1024**3),
            "max_allocated": torch.cuda.max_memory_allocated(device) / (1024**3),
            "max_reserved": torch.cuda.max_memory_reserved(device) / (1024**3),
        }

    @staticmethod
    def print_memory_stats(device: Optional[int] = None, prefix: str = ""):
        """Print memory statistics."""
        stats = CUDAMemoryManager.get_memory_stats(device)
        device = device or torch.cuda.current_device()
        print(f"{prefix}GPU {device} Memory: "
              f"Allocated={stats['allocated']:.2f}GB, "
              f"Reserved={stats['reserved']:.2f}GB, "
              f"Max={stats['max_allocated']:.2f}GB")

    @staticmethod
    def clear_cache():
        """Clear CUDA cache."""
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    @staticmethod
    def reset_peak_stats(device: Optional[int] = None):
        """Reset peak memory statistics."""
        torch.cuda.reset_peak_memory_stats(device)

    @staticmethod
    @contextmanager
    def track_memory(name: str = "Operation", device: Optional[int] = None):
        """Context manager to track memory usage of an operation."""
        device = device or torch.cuda.current_device()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

        start_allocated = torch.cuda.memory_allocated(device)

        yield

        torch.cuda.synchronize()
        end_allocated = torch.cuda.memory_allocated(device)
        peak_allocated = torch.cuda.max_memory_allocated(device)

        delta = (end_allocated - start_allocated) / (1024**3)
        peak = (peak_allocated - start_allocated) / (1024**3)

        print(f"[{name}] Memory: Delta={delta:+.2f}GB, Peak={peak:.2f}GB")


def get_optimal_batch_size(
    model_memory_gb: float,
    sequence_length: int,
    gpu_memory_gb: float = 80.0,
    safety_margin: float = 0.85,
    bytes_per_param: int = 2,
) -> int:
    """Estimate optimal batch size based on available memory."""
    available = gpu_memory_gb * safety_margin

    # Rough estimate: activations scale with batch_size * seq_len * hidden * layers
    activation_memory_per_sample_gb = sequence_length * 768 * 16 * bytes_per_param / (1024**3)

    # Available = total - model - optimizer (2x model for Adam states)
    available_for_activations = available - model_memory_gb * 3

    if available_for_activations <= 0:
        return 1

    batch_size = int(available_for_activations / activation_memory_per_sample_gb)
    return max(1, batch_size)


def synchronize_all_gpus():
    """Synchronize all GPUs."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)


def get_compute_capability() -> Tuple[int, int]:
    """Get compute capability of current GPU."""
    if not torch.cuda.is_available():
        return (0, 0)
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return (props.major, props.minor)


def is_ampere_or_newer() -> bool:
    """Check if GPU is Ampere (8.0+) or newer."""
    return get_compute_capability() >= (8, 0)


def is_hopper() -> bool:
    """Check if GPU is Hopper (9.0, e.g., H100)."""
    return get_compute_capability() >= (9, 0)


def get_optimal_worker_count(num_gpus: int = 1, reserved_cores: int = 4) -> int:
    """
    Calculate optimal dataloader workers.

    For high-core systems (Xeon Platinum 8480+ with 52 cores), allows up to 24
    workers per GPU to maximize data loading throughput on NVMe/high-bandwidth storage.

    Args:
        num_gpus: Number of GPUs to distribute workers across
        reserved_cores: Cores to reserve for other processes

    Returns:
        Optimal number of workers per GPU
    """
    total_cores = psutil.cpu_count(logical=False) or 52
    available_cores = max(1, total_cores - reserved_cores)
    workers_per_gpu = max(1, available_cores // max(1, num_gpus))

    # Cap based on system size:
    # - Small systems (< 16 cores): cap at 8
    # - Medium systems (16-32 cores): cap at 16
    # - Large systems (> 32 cores, e.g., Xeon Platinum): cap at 24
    if total_cores > 32:
        max_workers = 24
    elif total_cores > 16:
        max_workers = 16
    else:
        max_workers = 8

    return min(max_workers, workers_per_gpu)


def get_available_ram_gb() -> float:
    """Get available system RAM in GB."""
    mem = psutil.virtual_memory()
    return mem.available / (1024**3)


def should_preload_dataset(min_ram_gb: float = 128.0) -> bool:
    """Check if dataset should be preloaded to RAM. Eliminates I/O bottlenecks on high-RAM systems."""
    available = get_available_ram_gb()
    return available >= min_ram_gb


def print_h100_optimization_status():
    """Print H100 optimization status (TF32, kernels, etc.)."""
    print("\n" + "=" * 60)
    print("OPTIMIZATION STATUS")
    print("=" * 60)

    print("\nTensor Core Settings:")
    print(f"  TF32 Matmul: {'Y' if torch.backends.cuda.matmul.allow_tf32 else 'N'}")
    print(f"  TF32 cuDNN: {'Y' if torch.backends.cudnn.allow_tf32 else 'N'}")

    print("\nKernel Versions:")
    print(f"  FlashAttention-2: v{flash_attn.__version__}")
    print(f"  Mamba-SSM: v{mamba_ssm.__version__}")

    workers = get_optimal_worker_count(torch.cuda.device_count() or 1)
    print(f"\nDataloader Settings:")
    print(f"  Recommended workers per GPU: {workers}")
    print(f"  Dataset preload recommended: {should_preload_dataset()}")

    print("=" * 60 + "\n")


def setup_training_environment(verbose: bool = True) -> HardwareInfo:
    """One-call setup for optimal training environment."""
    info = detect_hardware()

    if verbose:
        print_hardware_info(info)

    setup_h100_optimizations()

    if info.gpu_count > 1:
        setup_nccl_optimizations(
            nvlink_available=info.nvlink_available,
            use_infiniband=False,
        )

    return info
