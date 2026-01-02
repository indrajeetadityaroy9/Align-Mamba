#!/usr/bin/env python3
"""
Migrate Legacy Checkpoints to Publication-Standard Format.

Converts legacy checkpoints (plain state_dict) to publication-standard format
with embedded config following NeurIPS/ICML/AISTATS reproducibility guidelines.

Usage:
    # Migrate Mamba checkpoint
    python scripts/migrate_checkpoint.py \
        --input outputs/thesis_fast/mamba/checkpoint-20000/model.pt \
        --output outputs/thesis_fast/mamba/checkpoint-20000/model_v2.pt \
        --model-type mamba

    # Migrate Transformer checkpoint
    python scripts/migrate_checkpoint.py \
        --input outputs/thesis_fast/transformer/checkpoint-35000/model.pt \
        --output outputs/thesis_fast/transformer/checkpoint-35000/model_v2.pt \
        --model-type transformer

    # Migrate all checkpoints in a directory
    python scripts/migrate_checkpoint.py --migrate-all
"""

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.encoder_decoder import ModelConfig


# Configurations matching the training runs
# NOTE: decoder_layers is the BASE layer count before adding cross-attention
# With cross_attn_every=4, layers are added at positions 4, 8, 12 (0-indexed after 3,7,11)
# So 12 base + 3 cross-attn = 15 total decoder layers
MAMBA_CONFIG = ModelConfig(
    vocab_size=32768,
    d_model=512,
    encoder_layers=12,
    decoder_layers=12,  # Base layers (15 total with cross-attn)
    d_state=64,
    d_conv=4,
    expand=2,
    n_heads=8,
    attention_ratio=0.125,  # 1:7 ratio (2 attention layers per 12)
    cross_attn_every=4,  # Cross-attn after layers 3,7,11 → indices 4,9,14
    dropout=0.0,  # Inference mode
    max_seq_len=8192,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
)

# Transformer: all attention, same cross-attn pattern
TRANSFORMER_CONFIG = ModelConfig(
    vocab_size=32768,
    d_model=512,
    encoder_layers=12,
    decoder_layers=12,  # Base layers (15 total with cross-attn)
    d_state=64,  # Not used for transformer
    d_conv=4,
    expand=2,
    n_heads=8,
    attention_ratio=1.0,  # Pure transformer
    cross_attn_every=4,  # Same pattern as Mamba
    dropout=0.0,  # Inference mode
    max_seq_len=8192,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
)


def migrate_checkpoint(
    input_path: str,
    output_path: str,
    config: ModelConfig,
) -> None:
    """
    Migrate a legacy checkpoint to publication-standard format.

    Args:
        input_path: Path to legacy checkpoint
        output_path: Path for migrated checkpoint
        config: ModelConfig for this checkpoint
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    print(f"Migrating: {input_path}")

    # Load legacy checkpoint
    checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)

    # Handle different legacy formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            # Plain state_dict
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Create publication-standard checkpoint
    new_checkpoint = {
        'config': asdict(config),  # REQUIRED
        'model_state_dict': state_dict,
        'global_step': 0,
        'epoch': 0,
        'best_metric': None,
        'metadata': {
            'pytorch_version': torch.__version__,
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'timestamp': datetime.now().isoformat(),
            'world_size': 1,
            'migrated_from': str(input_path),
        }
    }

    # Save migrated checkpoint
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_checkpoint, output_path)

    print(f"  -> Saved: {output_path}")
    print(f"     Config: d_model={config.d_model}, layers={config.encoder_layers}/{config.decoder_layers}")
    print(f"     Attention ratio: {config.attention_ratio}")


def migrate_all_checkpoints():
    """Migrate all known legacy checkpoints."""
    base = Path("doc_nmt_mamba/outputs/thesis_fast")

    migrations = [
        # Mamba checkpoints
        (base / "mamba/checkpoint-20000/model.pt", MAMBA_CONFIG),
        # Transformer checkpoints
        (base / "transformer/checkpoint-30000/model.pt", TRANSFORMER_CONFIG),
        (base / "transformer/checkpoint-35000/model.pt", TRANSFORMER_CONFIG),
    ]

    for input_path, config in migrations:
        if input_path.exists():
            output_path = input_path.with_suffix('.v2.pt')
            migrate_checkpoint(str(input_path), str(output_path), config)
        else:
            print(f"Skipping (not found): {input_path}")


def main():
    parser = argparse.ArgumentParser(description="Migrate legacy checkpoints")
    parser.add_argument("--input", type=str, help="Input checkpoint path")
    parser.add_argument("--output", type=str, help="Output checkpoint path")
    parser.add_argument("--model-type", type=str, choices=["mamba", "transformer"],
                        help="Model type (determines config)")
    parser.add_argument("--migrate-all", action="store_true",
                        help="Migrate all known checkpoints")

    args = parser.parse_args()

    if args.migrate_all:
        migrate_all_checkpoints()
    elif args.input and args.output and args.model_type:
        config = MAMBA_CONFIG if args.model_type == "mamba" else TRANSFORMER_CONFIG
        migrate_checkpoint(args.input, args.output, config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
