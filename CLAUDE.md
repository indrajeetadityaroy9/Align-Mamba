# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Polar-Mem-Mamba investigates state capacity limits in Selective SSMs (Mamba). It implements three mechanisms to overcome the "capacity cliff":

1. **Polarized channels** (A=0/A=1 fusion) - mitigates intra-layer recency bias
2. **Memory pool** with learned gating - retains information across layers
3. **Cross-attention** - enables retrieval beyond fixed state capacity

## Commands

### Install
```bash
pip install -e .
```

### Training (distributed)
```bash
torchrun --nproc_per_node=N -m align_mamba.train
```

### Evaluation
```bash
# Standard evaluation
align-eval outputs/best

# Capacity cliff analysis
align-eval outputs/best --mode capacity_cliff
```

## Architecture

### Single SOTA Mode
The system operates with one architecture: full Polar-Mem-Mamba combining all three innovations. No configuration needed—SOTA defaults are hardcoded.

### Key Components

**PolarizedMemBlock** (decoder layer):
- RMSNorm → Mamba2
- Polarized fusion: `[mamba_out, zero_proj(h), cumsum(one_proj(h))]`
- Memory pool with learned `score_proj` and `out_gate` (no fixed thresholds)

**MemoryPool**:
- `score_proj`: Learned importance scoring (replaces tau1)
- `out_gate`: Learned retrieval gating (replaces tau2)
- Top-k selection for pool updates
- Attention-based retrieval

**Encoder**: Bidirectional Mamba with attention at layers n//2 and n-1

**Decoder**: PolarizedMemBlock layers with cross-attention at positions (0, 2)

### Key Files
- `config.py` - Minimal Config dataclass with SOTA defaults
- `model.py` - PolarizedMemBlock, MemoryPool, Encoder, Decoder
- `train.py` - Distributed training (DDP, cosine scheduler)
- `evaluate.py` - Token accuracy, perplexity, capacity cliff
- `data.py` - MQAR dataset generation
- `kernels/` - Triton kernels for RMSNorm and cross-entropy

## Design Principles

1. **No manual tuning** - All thresholds replaced with learned gates
2. **SOTA as default** - Single optimal configuration
3. **Principled training** - Standard AdamW, global gradient clipping, fixed regularization

## Token Constants
- PAD=0, BOS=1, EOS=2, SEP=3
- Keys: 10-4095, Values: 4096-8191
- Vocab size: 8192
