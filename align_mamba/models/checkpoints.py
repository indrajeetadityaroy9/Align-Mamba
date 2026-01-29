"""Checkpoint utilities with embedded config for reproducibility."""

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
    """Load checkpoint, returning (state_dict, config, metadata)."""
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            config_dict = checkpoint.get('config')
            if config_dict is None:
                raise ValueError(
                    "Checkpoint missing required 'config' field. "
                    "Checkpoints must include embedded ModelConfig for reproducibility."
                )
            config = ModelConfig(**config_dict)
            metadata = checkpoint.get('metadata')
            return state_dict, config, metadata

        if 'model' in checkpoint:
            state_dict = checkpoint['model']
            config_dict = checkpoint.get('config')
            if config_dict is None:
                raise ValueError(
                    "Checkpoint missing required 'config' field. "
                    "Checkpoints must include embedded ModelConfig for reproducibility."
                )
            config = ModelConfig(**config_dict)
            metadata = checkpoint.get('metadata')
            return state_dict, config, metadata

    raise ValueError(
        "Invalid checkpoint format. Expected dict with 'model_state_dict' or 'model' key and 'config' field."
    )


def load_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    device: str = 'cuda',
    dtype: torch.dtype = torch.bfloat16,
    strict: bool = True,
) -> Tuple[nn.Module, ModelConfig]:
    """Load model from checkpoint for evaluation."""
    state_dict, config, metadata = load_checkpoint(checkpoint_path, map_location='cpu')

    if metadata:
        print(f"Checkpoint metadata:")
        print(f"  PyTorch: {metadata.get('pytorch_version', 'unknown')}")
        print(f"  CUDA: {metadata.get('cuda_version', 'unknown')}")
        print(f"  Timestamp: {metadata.get('timestamp', 'unknown')}")

    model = HybridMambaEncoderDecoder(
        config=config,
        device=device,
        dtype=dtype,
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing or unexpected:
        def _summarize_keys(keys, limit=10):
            if len(keys) <= limit:
                return keys
            prefixes = {}
            for k in keys:
                prefix = k.split('.')[0] if '.' in k else k
                prefixes[prefix] = prefixes.get(prefix, 0) + 1
            summary = [f"{p}: {c} keys" for p, c in sorted(prefixes.items())]
            return summary[:5] + [f"...and {len(prefixes) - 5} more prefixes"] if len(prefixes) > 5 else summary

        if strict:
            msg = f"State dict mismatch:\n"
            if missing:
                msg += f"  Missing {len(missing)} keys: {_summarize_keys(missing)}\n"
            if unexpected:
                msg += f"  Unexpected {len(unexpected)} keys: {_summarize_keys(unexpected)}\n"
            msg += (
                "\nPossible causes:\n"
                "  - Architecture config mismatch (d_model, n_layers, hybrid_positions)\n"
                "  - Checkpoint from different model version\n"
                "  - DDP/FSDP wrapper mismatch\n"
                "\nUse strict=False to load anyway (may cause runtime errors)."
            )
            raise RuntimeError(msg)
        else:
            if missing:
                print(f"Warning: Missing {len(missing)} keys in state_dict: {_summarize_keys(missing)}")
            if unexpected:
                print(f"Warning: Unexpected {len(unexpected)} keys in state_dict: {_summarize_keys(unexpected)}")

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
    """Save checkpoint with embedded config for reproducibility."""
    if hasattr(model, 'config'):
        config = asdict(model.config)
    elif hasattr(model, 'module') and hasattr(model.module, 'config'):
        config = asdict(model.module.config)
    else:
        raise ValueError(
            "Model does not have config attribute. "
            "Publication-standard checkpoints require embedded config."
        )

    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    checkpoint = {
        'config': config,
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

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if scheduler is not None and hasattr(scheduler, 'state_dict'):
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)

    print(f"Saved checkpoint to {output_path}")
    print(f"  Config: d_model={config['d_model']}, layers={config['encoder_layers']}/{config['decoder_layers']}")
    print(f"  Step: {global_step}, Epoch: {epoch}")
