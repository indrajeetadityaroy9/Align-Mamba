# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Align-Mamba investigates state capacity limitations of Selective State Space Models (Mamba) and demonstrates how Hybrid Mamba-Attention architectures overcome these limitations. The core finding: pure Mamba decoders can only store ~d_state tokens (typically 64) of key-value associations. Strategic cross-attention placement, especially at Layer 0 (the "Blind Start" fix), enables retrieval beyond this capacity limit.

## Commands

```bash
# Install
pip install -e .

# Training (uses Hydra configuration)
python -m align_mamba.train                                    # Default config
python -m align_mamba.train experiment=01_proof_mqar_cliff     # Use experiment preset
python -m align_mamba.train model.d_state=128 data.mqar.num_pairs=256  # Override params

# Multi-seed sweep (Hydra multirun)
python -m align_mamba.train -m project.seed=42,1337,2024 model.hybrid_positions=[],[0,2]

# Evaluation
python evaluate.py checkpoint=<path>

# CLI entry points (after pip install)
align-train    # Equivalent to python -m align_mamba.train
align-eval     # Equivalent to python evaluate.py
```

## Architecture

**Encoder**: `HybridBiMambaEncoder` - Bidirectional Mamba with sparse attention at layers N/2 and N-1

**Decoder**: `HybridMambaDecoder` - Unidirectional Mamba with cross-attention at `hybrid_positions`
- Layer 0 cross-attention is critical ("Blind Start" fix)
- Additional layers computed adaptively from `d_state / log(num_pairs/d_state)`

**Key Classes**:
- `models/align_mamba.py`: `HybridBlock`, `HybridBiMambaEncoder`, `HybridMambaDecoder`
- `models/encoder_decoder.py`: `ModelConfig`, `HybridMambaEncoderDecoder` (main wrapper)
- `training/trainer.py`: `NMTTrainer` with distributed training support
- `training/adaptive.py`: Adaptive hyperparameter computation
- `data/mqar.py`: MQAR (Multi-Query Associative Recall) synthetic dataset

## Configuration System

Hydra configs in `configs/`:
- `config.yaml` - Root config that composes model/training/data
- `model/hybrid_small.yaml` - Default model (d_model=256, d_state=64, 4 decoder layers)
- `training/default.yaml` - Training params (batch_size=256, max_steps=100000)
- `data/mqar.yaml` - MQAR task params (num_pairs, num_queries, mode)
- `experiment/` - Pre-configured experiments (01_proof_mqar_cliff, 02_mechanism_ablation, 04_d_state_scaling)

Key parameters:
- `model.hybrid_positions`: List of decoder layers with cross-attention (e.g., `[0,2]`), or `null` for adaptive
- `model.d_state`: Mamba state dimension (creates capacity cliff at this value)
- `data.mode`: `seq2seq` (encoder-decoder) or `decoder_only` (pure Mamba)
- `data.mqar.num_pairs`: Number of key-value pairs to memorize

## Hardware Assumptions

Optimized for NVIDIA H100:
- BF16 precision (hardcoded, never FP16)
- `torch.compile` with `max-autotune` mode
- TF32 matmul enabled
- Requires mamba-ssm>=2.0 and flash-attn>=2.5 for CUDA kernels

## MQAR Task Format

```
Encoder: [BOS, k1, :, v1, k2, :, v2, ..., kN, :, vN, EOS]
Decoder: [BOS, k_query1, k_query2, ..., EOS]
Output:  Predict values for queried keys
```

Token ranges: keys [10, 4096), values [4096, 8192). See `constants.py` for all token IDs.
