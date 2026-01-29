# Align-Mamba: State Capacity Limits in Selective SSMs

This codebase investigates the **state capacity limitations** of Selective State Space Models (Mamba) and demonstrates how **Hybrid Mamba-Attention** architectures overcome these limitations through strategic cross-attention placement.

## Research Hypothesis

### 1. The State Capacity Problem

Pure Mamba decoders have limited "state capacity" - they can only reliably store and retrieve approximately **d_state tokens** (typically 64) worth of key-value associations. When the number of associations exceeds this limit, accuracy degrades sharply - we call this the **"Capacity Cliff"**.

### 2. The Solution: Hybrid Cross-Attention

Hybrid Mamba-Attention models solve this by offloading retrieval to cross-attention layers:
- The **encoder** (bidirectional Mamba) stores key-value pairs
- The **decoder** uses cross-attention at strategic positions to retrieve from the encoder
- This allows accurate retrieval far beyond d_state capacity

### 3. The "Blind Start" Problem

A critical finding: **Layer 0 cross-attention is essential**. Without it, the decoder starts "blind" - it cannot create contextualized queries for later cross-attention layers. Configurations with cross-attention only at later layers (e.g., `[8, 16]`) fail completely.

## Quick Start

```bash
# Install
pip install -e .

# Run capacity cliff experiment (Hybrid should succeed, Pure Mamba should fail)
python -m align_mamba.train experiment=01_proof_mqar_cliff \
    model.hybrid_positions=[0,2] data.mqar.num_pairs=128

# Run Blind Start ablation
python -m align_mamba.train -m experiment=02_mechanism_ablation \
    'model.hybrid_positions=[],[0],[0,2],[8,16]'
```

## Experiments

| Experiment | Config | Purpose |
|------------|--------|---------|
| **01_proof_mqar_cliff** | `experiment=01_proof_mqar_cliff` | Prove the capacity cliff: pure Mamba fails at `num_pairs > d_state`, hybrid succeeds |
| **02_mechanism_ablation** | `experiment=02_mechanism_ablation` | Prove Layer 0 is critical: `[8,16]` fails, `[0]` succeeds |
| **04_d_state_scaling** | `experiment=04_d_state_scaling` | Validate that capacity scales as O(d_state) |

### Expected Results (num_pairs=128, d_state=64)

| Configuration | Expected Accuracy |
|---------------|-------------------|
| Pure Mamba `[]` | ~0% (state overflow) |
| Layer 0 only `[0]` | ~85-90% |
| Minimal hybrid `[0,2]` | ~95% |
| No Layer 0 `[8,16]` | ~0% (proves Blind Start) |

## Terminology

| Term | Meaning |
|------|---------|
| **d_state** | Mamba's internal state dimension (default 64, creates capacity cliff) |
| **hybrid_positions** | Decoder layers with cross-attention (e.g., `[0, 2]`) |
| **num_pairs** | Number of key-value pairs in MQAR task |
| **seq2seq mode** | Encoder-decoder with cross-attention (tests offloading hypothesis) |
| **decoder_only mode** | Pure Mamba (tests raw state capacity) |

## MQAR Synthetic Task

The **Multi-Query Associative Recall (MQAR)** task is a controlled benchmark for state capacity:

```
Encoder: [BOS, k1, :, v1, k2, :, v2, ..., kN, :, vN, EOS]
Decoder: [BOS, k3, k7, ..., EOS]
Output:  Model must predict values v3, v7, ... for queried keys
```

When `num_pairs > d_state`, pure Mamba "overflows" and cannot recall values correctly.

## Architecture

```
Encoder (Bidirectional Mamba)         Decoder (Unidirectional Mamba + Cross-Attn)
┌─────────────────────────────┐      ┌─────────────────────────────────────────┐
│  BiMamba Block 0            │      │  HybridBlock 0 (Mamba + Cross-Attn)     │ ← "Blind Start" fix
│  BiMamba Block 1            │  →   │  MambaBlock 1                           │
│  Attention Block 2          │  →   │  HybridBlock 2 (Mamba + Cross-Attn)     │
│  BiMamba Block 3            │      │  MambaBlock 3                           │
└─────────────────────────────┘      └─────────────────────────────────────────┘
         ↓                                          ↓
   Key-Value Store                        Query + Retrieve → Logits
```

**HybridBlock**: Mamba runs first to create contextualized queries, then cross-attention retrieves from encoder.

## Key Files

| File | Purpose |
|------|---------|
| `align_mamba/models/align_mamba.py` | HybridBlock, HybridBiMambaEncoder, HybridMambaDecoder |
| `align_mamba/models/encoder_decoder.py` | ModelConfig, HybridMambaEncoderDecoder wrapper |
| `align_mamba/data/mqar.py` | MQAR dataset for state capacity testing |
| `align_mamba/training/trainer.py` | NMTTrainer with full training loop |
| `align_mamba/configs/experiment/` | Pre-configured experiments |

## Configuration

Uses [Hydra](https://hydra.cc/) for configuration composition:

```bash
# Default config
python -m align_mamba.train

# Override parameters
python -m align_mamba.train model.d_state=128 data.mqar.num_pairs=256

# Use experiment preset
python -m align_mamba.train experiment=01_proof_mqar_cliff

# Multi-seed sweep (Hydra multirun)
python -m align_mamba.train -m experiment=01_proof_mqar_cliff \
    project.seed=42,1337,2024 \
    model.hybrid_positions=[],[0,2]
```

## Hardware Requirements

Optimized for **NVIDIA H100 80GB**:
- BF16 precision (never FP16)
- `torch.compile` with `max-autotune`
- Fused optimizer, TF32 matmul
- Batch size 256 for single H100

## Dependencies

```
torch>=2.3.0
mamba-ssm>=2.0
flash-attn>=2.5
hydra-core>=1.3
causal-conv1d>=1.2
```

## License

[Add license information]

## Citation

[Add paper citation when available]
