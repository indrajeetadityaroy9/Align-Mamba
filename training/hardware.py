"""
Hardware acceleration utilities optimized for NVIDIA H100.

H100 Hopper architecture features:
- Compute capability 9.0
- TF32 Tensor Cores (19x faster than FP32)
- BF16 Tensor Cores
- 80GB HBM3 memory
- 3TB/s memory bandwidth
"""

import os
import torch
import torch.backends.cudnn as cudnn


def get_device_info() -> dict:
    """Get detailed GPU information."""
    if not torch.cuda.is_available():
        return {'device': 'cpu', 'name': 'CPU'}

    device = torch.device('cuda')
    props = torch.cuda.get_device_properties(0)

    return {
        'device': device,
        'name': props.name,
        'compute_capability': f"{props.major}.{props.minor}",
        'total_memory_gb': props.total_memory / 1e9,
        'is_hopper': props.major >= 9,  # H100 is compute capability 9.0
        'is_ampere': props.major >= 8,  # A100 is compute capability 8.0
    }


def configure_hardware(verbose: bool = True) -> torch.device:
    """
    Configure PyTorch for optimal H100 performance.

    Optimizations applied:
    1. TF32 for matrix multiplications (19x faster than FP32)
    2. BF16 for mixed precision (native H100 support)
    3. cuDNN autotuning for convolutions
    4. Memory-efficient attention settings
    5. CUDA graph-friendly settings

    Returns:
        Configured torch device
    """
    if not torch.cuda.is_available():
        if verbose:
            print("CUDA not available. Using CPU.")
        return torch.device('cpu')

    device = torch.device('cuda')
    info = get_device_info()

    if verbose:
        print(f"GPU: {info['name']}")
        print(f"Compute Capability: {info['compute_capability']}")
        print(f"Memory: {info['total_memory_gb']:.1f} GB")

    # === TF32 Optimization (H100/A100) ===
    # TF32 uses 19 bits (vs 32 for FP32) with same dynamic range
    # Provides ~3x speedup with minimal accuracy loss
    if info['is_ampere'] or info['is_hopper']:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if verbose:
            print("TF32: Enabled (Tensor Core acceleration)")

    # === cuDNN Benchmark ===
    # Autoselects fastest convolution algorithm for fixed input sizes
    cudnn.benchmark = True
    cudnn.deterministic = False  # Faster, but non-deterministic
    if verbose:
        print("cuDNN: Benchmark mode enabled")

    # === Memory Optimization ===
    # Use memory-efficient SDPA (Scaled Dot-Product Attention) backend
    # Note: This affects nn.MultiheadAttention, not our custom Bahdanau attention
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        if verbose:
            print("Flash SDPA: Enabled")

    # === CUDA Memory Settings ===
    # Expandable segments reduce fragmentation
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

    # === Garbage Collection ===
    # Reduce Python GC overhead during training
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    if verbose:
        print(f"Available Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print("-" * 40)

    return device


def get_optimal_batch_size(model_size_mb: float = 100, seq_len: int = 100) -> int:
    """
    Estimate optimal batch size for H100 80GB.

    Args:
        model_size_mb: Approximate model size in MB
        seq_len: Maximum sequence length

    Returns:
        Recommended batch size
    """
    if not torch.cuda.is_available():
        return 32

    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    # H100 80GB can handle very large batches
    if total_mem_gb > 70:
        return 128  # H100 80GB
    elif total_mem_gb > 35:
        return 64   # A100 40GB
    else:
        return 32   # Smaller GPUs


def get_optimal_workers() -> int:
    """Get optimal number of DataLoader workers."""
    cpu_count = os.cpu_count() or 4
    # Use ~75% of available CPUs for data loading
    return min(cpu_count * 3 // 4, 16)


def compile_model(model, mode: str = "reduce-overhead"):
    """
    Apply torch.compile to model for H100 optimization.

    Modes:
    - "default": Good balance of compile time and speedup
    - "reduce-overhead": Faster compile, good for RNNs
    - "max-autotune": Maximum speedup, longer compile time

    Args:
        model: PyTorch model to compile
        mode: Compilation mode

    Returns:
        Compiled model (or original if compilation fails)
    """
    if not torch.cuda.is_available():
        return model

    # Check if H100 (benefits most from compilation)
    info = get_device_info()
    if not info.get('is_hopper', False):
        print(f"Note: torch.compile works best on H100. Current GPU: {info['name']}")

    try:
        # Inductor backend is optimal for H100
        compiled = torch.compile(
            model,
            mode=mode,
            backend="inductor",
            options={
                "triton.cudagraphs": True,  # Enable CUDA graphs
                "max_autotune": mode == "max-autotune",
            }
        )
        print(f"torch.compile: Enabled (mode={mode})")
        return compiled
    except Exception as e:
        print(f"torch.compile failed: {e}. Using eager mode.")
        return model


def create_optimized_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool = True,
    collate_fn=None,
    drop_last: bool = True
):
    """
    Create DataLoader optimized for H100 training.

    Optimizations:
    - Pinned memory for faster CPU->GPU transfer
    - Persistent workers to avoid respawn overhead
    - Prefetch factor for overlapping data loading
    - Optimal worker count
    """
    from torch.utils.data import DataLoader

    num_workers = get_optimal_workers()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=drop_last,
    )


class GradientAccumulator:
    """
    Efficient gradient accumulation with automatic scaling.

    Handles:
    - Loss scaling for accumulation
    - Gradient clipping
    - Optimizer step synchronization
    """

    def __init__(
        self,
        optimizer,
        accumulation_steps: int = 4,
        max_grad_norm: float = 1.0
    ):
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.step_count = 0

    def backward(self, loss):
        """Backward pass with accumulation scaling."""
        scaled_loss = loss / self.accumulation_steps
        scaled_loss.backward()
        self.step_count += 1

    def step(self, model_params):
        """
        Perform optimizer step if accumulation complete.

        Returns:
            True if optimizer step was taken, False otherwise
        """
        if self.step_count % self.accumulation_steps == 0:
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model_params, self.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            return True
        return False

    def get_loss_scale(self) -> float:
        """Get the loss scaling factor for logging."""
        return self.accumulation_steps
