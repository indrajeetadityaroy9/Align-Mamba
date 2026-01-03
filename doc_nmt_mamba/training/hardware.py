"""
Hardware Detection and Optimization for H100 GPUs.

Provides:
- Automatic hardware detection (GPU, CPU, memory)
- H100-specific optimizations (TF32, BF16, NVLink, NCCL)
- CUDA memory management utilities
- Performance profiling helpers
- Optimal worker count calculation for high-CPU systems
"""

import os
import subprocess
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from contextlib import contextmanager
import warnings

import torch
import torch.distributed as dist

# psutil for cross-platform system info
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    warnings.warn("psutil not available. Install with: pip install psutil")


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
    # GPU
    gpu_count: int = 0
    gpus: List[GPUInfo] = field(default_factory=list)
    nvlink_available: bool = False
    nvlink_pairs: List[Tuple[int, int]] = field(default_factory=list)

    # CPU
    cpu_count: int = 0
    cpu_model: str = ""

    # Memory
    system_memory_gb: float = 0.0

    # CUDA
    cuda_version: str = ""
    cudnn_version: str = ""

    # Packages
    has_flash_attn: bool = False
    has_mamba_ssm: bool = False
    has_triton: bool = False

    # PyTorch
    torch_version: str = ""
    torch_compile_available: bool = False


def get_gpu_info(index: int) -> GPUInfo:
    """Get detailed information about a specific GPU."""
    props = torch.cuda.get_device_properties(index)

    total_memory = props.total_memory / (1024**3)
    free_memory = (props.total_memory - torch.cuda.memory_allocated(index)) / (1024**3)

    compute_cap = (props.major, props.minor)

    # H100 has compute capability 9.0
    is_h100 = compute_cap >= (9, 0) or "H100" in props.name

    # BF16 requires compute capability >= 8.0 (Ampere+)
    supports_bf16 = compute_cap >= (8, 0)

    # TF32 requires compute capability >= 8.0
    supports_tf32 = compute_cap >= (8, 0)

    # Flash Attention 2 works best on Ampere+ (8.0+)
    supports_flash_attn = compute_cap >= (8, 0)

    return GPUInfo(
        index=index,
        name=props.name,
        compute_capability=compute_cap,
        total_memory_gb=total_memory,
        free_memory_gb=free_memory,
        is_h100=is_h100,
        supports_bf16=supports_bf16,
        supports_tf32=supports_tf32,
        supports_flash_attn=supports_flash_attn,
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

    # PyTorch version
    info.torch_version = torch.__version__
    info.torch_compile_available = hasattr(torch, 'compile')

    # CUDA version
    if torch.cuda.is_available():
        info.cuda_version = torch.version.cuda or ""
        info.cudnn_version = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else ""

    # GPU detection
    info.gpu_count = torch.cuda.device_count()
    for i in range(info.gpu_count):
        info.gpus.append(get_gpu_info(i))

    # NVLink detection
    info.nvlink_available, info.nvlink_pairs = detect_nvlink()

    # CPU info
    info.cpu_count = os.cpu_count() or 1
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if 'model name' in line:
                    info.cpu_model = line.split(':')[1].strip()
                    break
    except (FileNotFoundError, PermissionError):
        info.cpu_model = "Unknown"

    # System memory
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if 'MemTotal' in line:
                    mem_kb = int(line.split()[1])
                    info.system_memory_gb = mem_kb / (1024**2)
                    break
    except (FileNotFoundError, PermissionError):
        info.system_memory_gb = 0.0

    # Check optional packages
    try:
        import flash_attn
        info.has_flash_attn = True
    except ImportError:
        info.has_flash_attn = False

    try:
        import mamba_ssm
        info.has_mamba_ssm = True
    except ImportError:
        info.has_mamba_ssm = False

    try:
        import triton
        info.has_triton = True
    except ImportError:
        info.has_triton = False

    return info


def print_hardware_info(info: Optional[HardwareInfo] = None):
    """Print formatted hardware information."""
    if info is None:
        info = detect_hardware()

    print("\n" + "=" * 70)
    print("HARDWARE CONFIGURATION")
    print("=" * 70)

    # PyTorch
    print(f"\nPyTorch: {info.torch_version}")
    print(f"CUDA: {info.cuda_version}")
    print(f"cuDNN: {info.cudnn_version}")
    print(f"torch.compile: {'✓' if info.torch_compile_available else '✗'}")

    # GPUs
    print(f"\n{'─' * 70}")
    print(f"GPUs: {info.gpu_count}")
    for gpu in info.gpus:
        h100_tag = " [H100]" if gpu.is_h100 else ""
        print(f"  [{gpu.index}] {gpu.name}{h100_tag}")
        print(f"      Memory: {gpu.free_memory_gb:.1f} / {gpu.total_memory_gb:.1f} GB")
        print(f"      Compute: {gpu.compute_capability[0]}.{gpu.compute_capability[1]}")
        print(f"      BF16: {'✓' if gpu.supports_bf16 else '✗'} | "
              f"TF32: {'✓' if gpu.supports_tf32 else '✗'} | "
              f"FlashAttn: {'✓' if gpu.supports_flash_attn else '✗'}")

    # NVLink
    if info.gpu_count > 1:
        print(f"\n{'─' * 70}")
        print(f"NVLink: {'✓ Available' if info.nvlink_available else '✗ Not available'}")
        if info.nvlink_pairs:
            pairs_str = ", ".join([f"{i}↔{j}" for i, j in info.nvlink_pairs if i < j])
            print(f"  Pairs: {pairs_str}")

    # CPU & Memory
    print(f"\n{'─' * 70}")
    print(f"CPU: {info.cpu_model}")
    print(f"CPU Cores: {info.cpu_count}")
    print(f"System Memory: {info.system_memory_gb:.1f} GB")

    # Optional packages
    print(f"\n{'─' * 70}")
    print("Optional Packages:")
    print(f"  flash-attn: {'✓' if info.has_flash_attn else '✗ (using PyTorch SDPA fallback)'}")
    print(f"  mamba-ssm: {'✓' if info.has_mamba_ssm else '✗ CRITICAL - Install with: pip install mamba-ssm'}")
    print(f"  triton: {'✓' if info.has_triton else '✗'}")

    print("=" * 70 + "\n")

    # Warnings
    if not info.has_mamba_ssm:
        warnings.warn(
            "mamba-ssm not installed! Mamba-2 CUDA kernels will not work. "
            "Install with: pip install mamba-ssm causal-conv1d --no-build-isolation"
        )


def setup_h100_optimizations(
    enable_tf32: bool = True,
    enable_cudnn_benchmark: bool = True,
    enable_flash_sdp: bool = True,
    enable_mem_efficient_sdp: bool = True,
    cuda_alloc_conf: Optional[str] = None,
    matmul_precision: str = "high",
) -> Dict[str, bool]:
    """
    Apply H100-specific optimizations for maximum throughput.

    H100 (Hopper) specific optimizations:
    - TF32/BF16 for Tensor Core acceleration (~3-5x speedup vs FP32)
    - Flash SDP for memory-efficient attention
    - High precision matmul (BF16 accumulation)
    - Optimized CUDA memory allocator

    Args:
        enable_tf32: Enable TF32 for matmul (faster, slight precision loss)
        enable_cudnn_benchmark: Enable cuDNN benchmark mode
        enable_flash_sdp: Enable Flash SDP for attention
        enable_mem_efficient_sdp: Enable memory-efficient SDP
        cuda_alloc_conf: Custom CUDA allocator config
        matmul_precision: Matmul precision ('highest', 'high', 'medium')
                          'high' = BF16 accumulation, 'medium' = TF32

    Returns:
        Dict of applied optimizations
    """
    applied = {}

    # 1. TF32 for faster matmul on Ampere+ GPUs
    # H100 gets ~3x speedup on FP32 math with TF32 enabled
    if enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        applied["tf32"] = True

    # 2. Set matmul precision for H100 Tensor Cores
    # 'high' = BF16 accumulation (best for H100)
    # 'medium' = TF32 (good balance)
    # 'highest' = FP32 (slowest, most precise)
    torch.set_float32_matmul_precision(matmul_precision)
    applied["matmul_precision"] = matmul_precision

    # 3. cuDNN benchmark for optimized convolutions
    if enable_cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        applied["cudnn_benchmark"] = True

    # 4. Flash SDP (Scaled Dot Product Attention)
    if enable_flash_sdp:
        torch.backends.cuda.enable_flash_sdp(True)
        applied["flash_sdp"] = True

    if enable_mem_efficient_sdp:
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        applied["mem_efficient_sdp"] = True

    # 5. CUDA memory allocator configuration
    # expandable_segments reduces fragmentation for large models
    if cuda_alloc_conf is None:
        cuda_alloc_conf = "expandable_segments:True"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = cuda_alloc_conf
    applied["cuda_alloc_conf"] = cuda_alloc_conf

    # 6. Check FlashAttention-2 availability (critical for H100 performance)
    try:
        from flash_attn import flash_attn_func
        applied["flash_attn_2"] = True
        print("FlashAttention-2 is available and optimized for H100.")
    except ImportError:
        applied["flash_attn_2"] = False
        print("WARNING: FlashAttention-2 not found. Using PyTorch SDPA fallback.")

    return applied


def get_optimal_compiler_backend() -> str:
    """
    Get optimal torch.compile backend for current hardware.

    H100 benefits massively from TorchInductor with cudagraphs.

    Returns:
        Backend name for torch.compile
    """
    if is_hopper():
        return "inductor"  # Best for H100 with cudagraphs
    elif is_ampere_or_newer():
        return "inductor"  # Good for A100/A10
    else:
        return "inductor"  # Default to inductor


def get_optimal_compile_options() -> Dict[str, Any]:
    """
    Get optimal torch.compile options for H100.

    Returns:
        Dict of compile options
    """
    options = {
        "mode": "max-autotune",  # Aggressive optimization
        "fullgraph": False,      # Allow graph breaks for compatibility
    }

    if is_hopper():
        # H100-specific optimizations
        options["options"] = {
            "triton.cudagraphs": True,  # Enable CUDA graphs for reduced overhead
            "epilogue_fusion": True,     # Fuse epilogue operations
            "max_autotune": True,        # More extensive autotuning
        }

    return options


def setup_nccl_optimizations(
    nvlink_available: bool = True,
    use_infiniband: bool = False,
) -> Dict[str, str]:
    """
    Setup NCCL optimizations for multi-GPU training.

    Args:
        nvlink_available: Whether NVLink is available
        use_infiniband: Whether to use InfiniBand

    Returns:
        Dict of environment variables set
    """
    env_vars = {}

    if nvlink_available:
        # Use NVLink for peer-to-peer transfers
        os.environ["NCCL_P2P_LEVEL"] = "NVL"
        env_vars["NCCL_P2P_LEVEL"] = "NVL"

        # Enable P2P
        os.environ["NCCL_P2P_DISABLE"] = "0"
        env_vars["NCCL_P2P_DISABLE"] = "0"

    if use_infiniband:
        os.environ["NCCL_IB_DISABLE"] = "0"
        env_vars["NCCL_IB_DISABLE"] = "0"
    else:
        os.environ["NCCL_IB_DISABLE"] = "1"
        env_vars["NCCL_IB_DISABLE"] = "1"

    # NCCL debug (set to WARN for less verbose, INFO for debugging)
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    env_vars["NCCL_DEBUG"] = os.environ["NCCL_DEBUG"]

    # Use all available NICs for NCCL
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
    bytes_per_param: int = 2,  # BF16
) -> int:
    """
    Estimate optimal batch size based on available memory.

    Args:
        model_memory_gb: Model size in GB
        sequence_length: Sequence length
        gpu_memory_gb: Available GPU memory
        safety_margin: Fraction of memory to use
        bytes_per_param: Bytes per parameter (2 for BF16, 4 for FP32)

    Returns:
        Estimated optimal batch size
    """
    available = gpu_memory_gb * safety_margin

    # Rough estimate: activations scale with batch_size * seq_len
    # Memory per sample ≈ seq_len * hidden_dim * num_layers * bytes_per_param
    # This is a very rough estimate
    activation_memory_per_sample_gb = sequence_length * 768 * 16 * bytes_per_param / (1024**3)

    # Available for activations = total - model - optimizer (2x model for Adam)
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
    """Check if GPU is Ampere (compute capability 8.0) or newer."""
    return get_compute_capability() >= (8, 0)


def is_hopper() -> bool:
    """Check if GPU is Hopper (compute capability 9.0, e.g., H100)."""
    return get_compute_capability() >= (9, 0)


def get_optimal_worker_count(num_gpus: int = 1, reserved_cores: int = 4) -> int:
    """
    Calculate optimal dataloader workers based on available CPU cores.

    For 52 vCPU systems, this calculates workers per GPU while leaving
    cores for OS/overhead.

    Args:
        num_gpus: Number of GPUs (workers are divided among GPUs)
        reserved_cores: Cores to reserve for OS/overhead

    Returns:
        Optimal number of workers per GPU (capped at 16)
    """
    if PSUTIL_AVAILABLE:
        # Use physical cores (not logical/hyperthreaded)
        total_cores = psutil.cpu_count(logical=False) or 52
    else:
        total_cores = os.cpu_count() or 52

    available_cores = max(1, total_cores - reserved_cores)
    workers_per_gpu = max(1, available_cores // max(1, num_gpus))

    # Cap at 16 to prevent diminishing returns and overhead
    return min(16, workers_per_gpu)


def get_available_ram_gb() -> float:
    """
    Get available system RAM in GB.

    Returns:
        Available RAM in GB, or 0 if detection fails
    """
    if PSUTIL_AVAILABLE:
        mem = psutil.virtual_memory()
        return mem.available / (1024**3)
    else:
        # Fallback to /proc/meminfo on Linux
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemAvailable' in line:
                        mem_kb = int(line.split()[1])
                        return mem_kb / (1024**2)
        except (FileNotFoundError, PermissionError):
            pass
        return 0.0


def should_preload_dataset(min_ram_gb: float = 128.0) -> bool:
    """
    Check if dataset should be preloaded to RAM.

    For 450GB RAM systems, preloading eliminates I/O bottlenecks.

    Args:
        min_ram_gb: Minimum free RAM required for preloading

    Returns:
        True if enough RAM is available for preloading
    """
    available = get_available_ram_gb()
    return available >= min_ram_gb


def print_h100_optimization_status():
    """
    Print comprehensive H100 optimization status.

    Call at training start to verify optimal configuration.
    """
    print("\n" + "=" * 60)
    print("H100 OPTIMIZATION STATUS")
    print("=" * 60)

    # GPU info
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            is_h100 = props.major >= 9 or "H100" in props.name
            tag = " [H100]" if is_h100 else ""
            print(f"GPU {i}: {props.name}{tag}")
            print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
            print(f"  Compute: {props.major}.{props.minor}")

    # CPU/RAM info
    workers = get_optimal_worker_count(torch.cuda.device_count() or 1)
    ram_gb = get_available_ram_gb()
    print(f"\nCPU Workers per GPU: {workers}")
    print(f"Available RAM: {ram_gb:.1f} GB")
    print(f"Dataset preload recommended: {should_preload_dataset()}")

    # Kernel status
    print("\nKernel Status:")
    try:
        import flash_attn
        print(f"  FlashAttention-2: v{flash_attn.__version__}")
    except ImportError:
        print("  FlashAttention-2: NOT INSTALLED")

    try:
        import mamba_ssm
        print(f"  Mamba-SSM: v{mamba_ssm.__version__}")
    except ImportError:
        print("  Mamba-SSM: NOT INSTALLED (CRITICAL)")

    # TF32 status
    print(f"\nTF32 Matmul: {torch.backends.cuda.matmul.allow_tf32}")
    print(f"TF32 cuDNN: {torch.backends.cudnn.allow_tf32}")

    print("=" * 60 + "\n")


# Quick setup function for training
def setup_training_environment(verbose: bool = True) -> HardwareInfo:
    """
    One-call setup for optimal training environment.

    Args:
        verbose: Print hardware info

    Returns:
        HardwareInfo object
    """
    # Detect hardware
    info = detect_hardware()

    if verbose:
        print_hardware_info(info)

    # Apply H100 optimizations
    has_ampere = any(gpu.compute_capability >= (8, 0) for gpu in info.gpus)

    setup_h100_optimizations(
        enable_tf32=has_ampere,
        enable_cudnn_benchmark=True,
        enable_flash_sdp=info.has_flash_attn or has_ampere,
        enable_mem_efficient_sdp=has_ampere,
    )

    # Apply NCCL optimizations
    if info.gpu_count > 1:
        setup_nccl_optimizations(
            nvlink_available=info.nvlink_available,
            use_infiniband=False,
        )

    # Warn about missing packages
    if not info.has_mamba_ssm:
        print("\n⚠️  WARNING: mamba-ssm not installed!")
        print("   Install with: pip install mamba-ssm causal-conv1d --no-build-isolation")
        print("   Without this, Mamba-2 CUDA kernels will NOT work.\n")

    return info
