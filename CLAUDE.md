# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Align-Mamba investigates state capacity limitations in Selective State Space Models (Mamba) and demonstrates how Hybrid Mamba-Attention architectures solve these limitations through strategic cross-attention placement.

**Core Research Hypothesis:**
- Pure Mamba decoders have limited "state capacity" (~d_state tokens, typically 64)
- When key-value associations exceed this limit, accuracy degrades sharply ("Capacity Cliff")
- Hybrid models fix this by placing cross-attention at strategic decoder positions
- Layer 0 cross-attention is critical ("Blind Start" fix) - without it, decoders cannot create contextualized queries

## Commands

### Installation
```bash
pip install -e .
```

### Training
```bash
# Single GPU
python train.py

# Multi-GPU (recommended)
python scripts/launch.py

# Multi-GPU with config overrides
python scripts/launch.py model.d_model=512 training.max_steps=10000

# Run specific experiment preset
python train.py experiment=01_proof_mqar_cliff

# Override parameters
python train.py model.d_state=128 data.num_pairs=256

# Multi-seed sweep (Hydra multirun)
python train.py -m experiment=01_proof_mqar_cliff \
    project.seed=42,1337,2024 'model.hybrid_positions=[],[0,2]'
```

### Evaluation
```bash
python evaluate.py checkpoint=outputs/<run>/model.pt data.num_pairs=128
```

## Architecture

### Module Structure
```
./
├── train.py              # Training entry point
├── evaluate.py           # Evaluation entry point
├── constants.py          # Hardcoded infrastructure values (token IDs, vocab ranges, H100 settings)
├── models/               # Core architecture
│   ├── encoder_decoder.py   # ModelConfig, HybridMambaEncoderDecoder (main wrapper)
│   ├── align_mamba.py       # HybridBiMambaEncoder, HybridMambaDecoder
│   ├── attention.py         # BidirectionalAttention, FlashCrossAttention
│   └── wrapper.py           # Mamba2BlockWrapper around official CUDA kernels
├── training/             # Training infrastructure
│   ├── trainer.py           # NMTTrainer with full training loop
│   ├── distributed.py       # Multi-GPU DDP setup
│   └── objectives.py        # LabelSmoothingCrossEntropy, CosineAnnealingWarmupScheduler
├── data/                 # Data pipeline
│   ├── mqar.py              # MQARDataset, MQARConfig (synthetic task)
│   └── collator.py          # MQARCollator for batching
└── configs/              # Hydra YAML configuration
    ├── config.yaml          # Main config with defaults composition
    ├── model/               # Model presets (hybrid_small.yaml)
    ├── training/            # Training presets
    ├── data/                # Data presets (mqar.yaml)
    └── experiment/          # Pre-configured experiment sweeps
```

### Key Design Patterns

**Hybrid Positions:** Computed via `compute_hybrid_positions(n_layers)` → returns `[0, N//3, 2N//3]`. Layer 0 is always included to fix the "Blind Start" problem. Override via `model.hybrid_positions` in config.

**Decoder Architecture:** All decoder layers are Mamba blocks (`self.layers`). Cross-attention is added at hybrid positions (`self.cross_attn` ModuleDict). Mamba runs first to produce contextualized queries, then cross-attention retrieves from encoder states.

**MQAR Synthetic Task:** Tests associative recall capacity.
- Encoder receives: `[BOS, k1, :, v1, ..., kN, :, vN, EOS]` (key-value pairs)
- Decoder receives: `[BOS, k3, k7, ..., EOS]` (queries)
- Model must predict: Values `v3, v7, ...` for queried keys

Two modes: `seq2seq` (encoder-decoder) and `decoder_only` (pure Mamba state capacity test).

### Configuration System

Hydra-based configuration with defaults composition:
```yaml
defaults:
  - model: hybrid_small      # Model architecture
  - training: default        # Training hyperparameters
  - data: mqar              # Dataset configuration
  - optional experiment: null # Experiment preset
```

**Experiment Presets:**
- `01_proof_mqar_cliff` - Prove capacity cliff (pure Mamba fails at num_pairs > d_state)
- `02_mechanism_ablation` - Prove Layer 0 criticality
- `04_d_state_scaling` - Validate capacity scales as O(d_state)

## Key Implementation Details

- **BF16-only:** Never uses FP16 (H100 architectural choice)
- **Official mamba-ssm:** Uses official CUDA kernels via `mamba-ssm >= 2.0`
- **Gradient Checkpointing:** Optional via `model.gradient_checkpointing_enable()`
- **Adaptive Gradient Clipping (AGC):** NFNet-style for training stability
- **DecoderInferenceCache:** O(1) cached decoding for Mamba states

## Dependencies

Core requirements:
- `torch >= 2.3.0` (BF16 native support)
- `mamba-ssm >= 2.0` (Official SSM with CUDA kernels)
- `causal-conv1d >= 1.2` (Efficient convolution for Mamba)
- `flash-attn >= 2.5` (Fast attention backend)
- `hydra-core >= 1.3` (Configuration management)
