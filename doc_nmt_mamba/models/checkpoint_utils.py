"""
Checkpoint Utilities for Hybrid Mamba-Attention NMT.

Publication-standard checkpoint format following NeurIPS/ICML/AISTATS guidelines:
- Config ALWAYS embedded (required for reproducibility)
- Full metadata for environment tracking
- Single unified format (no legacy handling)

Checkpoint Format:
    {
        'config': dict,  # ModelConfig as dict (REQUIRED)
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer_state,  # Optional
        'scheduler_state_dict': scheduler_state,  # Optional
        'global_step': int,
        'epoch': int,
        'best_metric': float,
        'metadata': {
            'pytorch_version': str,
            'cuda_version': str,
            'timestamp': str,
            'world_size': int,
        }
    }
"""

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from .encoder_decoder import ModelConfig, HybridMambaEncoderDecoder


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    map_location: str = 'cpu',
) -> Tuple[Dict, ModelConfig, Optional[Dict]]:
    """
    Load checkpoint file and extract components.

    Args:
        checkpoint_path: Path to the checkpoint file
        map_location: Device to load tensors to

    Returns:
        Tuple of (state_dict, config, metadata)

    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        ValueError: If checkpoint missing required 'config' field
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

    # Handle checkpoint format
    if isinstance(checkpoint, dict):
        # New format with model_state_dict key
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            config_dict = checkpoint.get('config')
            if config_dict is None:
                raise ValueError(
                    f"Checkpoint missing 'config' field. "
                    f"Publication-standard checkpoints must include embedded config. "
                    f"Use scripts/migrate_checkpoint.py to add config to legacy checkpoints."
                )
            config = ModelConfig(**config_dict)
            metadata = checkpoint.get('metadata')
            return state_dict, config, metadata

        # Alternative format with 'model' key
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
            config_dict = checkpoint.get('config')
            if config_dict is None:
                raise ValueError(
                    f"Checkpoint missing 'config' field. "
                    f"Publication-standard checkpoints must include embedded config. "
                    f"Use scripts/migrate_checkpoint.py to add config to legacy checkpoints."
                )
            config = ModelConfig(**config_dict)
            metadata = checkpoint.get('metadata')
            return state_dict, config, metadata

    # Legacy format (plain state_dict) - not supported
    raise ValueError(
        f"Legacy checkpoint format detected (plain state_dict without config). "
        f"Publication-standard checkpoints must include embedded config. "
        f"Use scripts/migrate_checkpoint.py to convert legacy checkpoints."
    )


def load_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    device: str = 'cuda',
    dtype: torch.dtype = torch.bfloat16,
    strict: bool = True,
) -> Tuple[nn.Module, ModelConfig]:
    """
    Load model from checkpoint.

    This is the primary entry point for loading models for evaluation.
    Requires publication-standard checkpoint format with embedded config.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load model on ('cuda' or 'cpu')
        dtype: Data type for model parameters
        strict: Whether to require exact state_dict match

    Returns:
        Tuple of (model, config)

    Example:
        >>> model, config = load_model_from_checkpoint("outputs/best_model.pt")
        >>> model.eval()
        >>> outputs = model.generate(src_ids)
    """
    # Load checkpoint components
    state_dict, config, metadata = load_checkpoint(checkpoint_path, map_location='cpu')

    # Print metadata if available
    if metadata:
        print(f"Checkpoint metadata:")
        print(f"  PyTorch: {metadata.get('pytorch_version', 'unknown')}")
        print(f"  CUDA: {metadata.get('cuda_version', 'unknown')}")
        print(f"  Timestamp: {metadata.get('timestamp', 'unknown')}")

    # Build model with loaded config
    model = HybridMambaEncoderDecoder(
        config=config,
        device=device,
        dtype=dtype,
    )

    # Load weights
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing or unexpected:
        if strict:
            raise RuntimeError(
                f"State dict mismatch - Missing: {len(missing)}, Unexpected: {len(unexpected)}. "
                f"Missing keys: {missing[:5]}... "
                f"Unexpected keys: {unexpected[:5]}... "
                f"Use strict=False to load anyway."
            )
        else:
            if missing:
                print(f"Warning: Missing keys in state_dict: {missing[:5]}...")
            if unexpected:
                print(f"Warning: Unexpected keys in state_dict: {unexpected[:5]}...")

    model = model.to(device)
    model.eval()

    return model, config


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[object] = None,
    global_step: int = 0,
    epoch: int = 0,
    best_metric: Optional[float] = None,
    output_path: Union[str, Path] = "checkpoint.pt",
    world_size: int = 1,
) -> None:
    """
    Save checkpoint in publication-standard format.

    Follows NeurIPS/ICML reproducibility guidelines by including:
    - Model state dict
    - ModelConfig for exact architecture reproduction (REQUIRED)
    - Optimizer/scheduler state for training resumption
    - Metadata for environment tracking

    Args:
        model: The model to save
        optimizer: Optional optimizer for training resumption
        scheduler: Optional scheduler for training resumption
        global_step: Current training step
        epoch: Current epoch
        best_metric: Best validation metric achieved
        output_path: Path to save checkpoint
        world_size: Number of distributed processes

    Raises:
        ValueError: If model doesn't have config attribute
    """
    # Get config from model (REQUIRED)
    if hasattr(model, 'config'):
        config = asdict(model.config)
    elif hasattr(model, 'module') and hasattr(model.module, 'config'):
        # Handle DDP wrapped models
        config = asdict(model.module.config)
    else:
        raise ValueError(
            "Model does not have config attribute. "
            "Publication-standard checkpoints require embedded config."
        )

    # Get state dict (handle DDP)
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    # Build checkpoint
    checkpoint = {
        'config': config,  # REQUIRED
        'model_state_dict': state_dict,
        'global_step': global_step,
        'epoch': epoch,
        'best_metric': best_metric,
        'metadata': {
            'pytorch_version': torch.__version__,
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'timestamp': datetime.now().isoformat(),
            'world_size': world_size,
        }
    }

    # Add optimizer state if provided
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    # Add scheduler state if provided
    if scheduler is not None and hasattr(scheduler, 'state_dict'):
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)

    print(f"Saved checkpoint to {output_path}")
    print(f"  Config: d_model={config['d_model']}, layers={config['encoder_layers']}/{config['decoder_layers']}")
    print(f"  Step: {global_step}, Epoch: {epoch}")
