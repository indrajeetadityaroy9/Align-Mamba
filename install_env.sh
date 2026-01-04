#!/bin/bash
# =============================================================================
# Align-Mamba: H100 Environment Setup Script
# =============================================================================
#
# ICML/NeurIPS Reproducibility Requirements:
# - PyTorch must be installed BEFORE Mamba/FlashAttention (CUDA header linking)
# - --no-build-isolation is CRITICAL for kernel compilation
# - SpaCy models needed for Entity Recall evaluation
#
# Target: PyTorch 2.3.0 + CUDA 12.1 (H100 native support)
#
# Usage:
#   chmod +x install_env.sh
#   ./install_env.sh
#
# =============================================================================

set -e  # Exit on error

echo "=============================================="
echo "Align-Mamba H100 Environment Setup"
echo "=============================================="

# Check if we're in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "WARNING: No virtual environment detected."
    echo "Consider running: python -m venv venv && source venv/bin/activate"
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# =============================================================================
# Step 1: Install PyTorch (H100 requires CUDA 11.8 or 12.1+)
# =============================================================================
echo ""
echo "[1/5] Installing PyTorch 2.3.0 + CUDA 12.1..."
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch CUDA
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# =============================================================================
# Step 2: Install Build Tools (Required for compiling kernels)
# =============================================================================
echo ""
echo "[2/5] Installing build tools..."
pip install packaging ninja wheel

# =============================================================================
# Step 3: Install Optimized Kernels (Order matters!)
# =============================================================================
# CRITICAL: --no-build-isolation ensures kernels see the installed PyTorch
echo ""
echo "[3/5] Installing optimized CUDA kernels..."
echo "      This may take 5-10 minutes for compilation..."

# Mamba-2 SSM kernels
echo "  -> Installing mamba-ssm..."
pip install mamba-ssm>=2.0.0 --no-build-isolation

# Causal convolution kernels (Mamba dependency)
echo "  -> Installing causal-conv1d..."
pip install causal-conv1d>=1.2.0 --no-build-isolation

# FlashAttention-2 (H100 optimized)
echo "  -> Installing flash-attn..."
pip install flash-attn>=2.5.8 --no-build-isolation

# =============================================================================
# Step 4: Install Project Requirements
# =============================================================================
echo ""
echo "[4/5] Installing project requirements..."
pip install -r requirements.txt

# Install the project in editable mode
pip install -e .

# =============================================================================
# Step 5: Download SpaCy Models (For Entity Recall evaluation)
# =============================================================================
echo ""
echo "[5/5] Downloading SpaCy language models..."
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm

# =============================================================================
# Verification
# =============================================================================
echo ""
echo "=============================================="
echo "Verifying installation..."
echo "=============================================="

python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')

try:
    import mamba_ssm
    print(f'Mamba-SSM: OK')
except ImportError as e:
    print(f'Mamba-SSM: FAILED ({e})')

try:
    import flash_attn
    print(f'FlashAttention: {flash_attn.__version__}')
except ImportError as e:
    print(f'FlashAttention: FAILED ({e})')

try:
    import spacy
    nlp_en = spacy.load('en_core_web_sm')
    nlp_de = spacy.load('de_core_news_sm')
    print(f'SpaCy models: OK')
except Exception as e:
    print(f'SpaCy models: FAILED ({e})')

print()
print('Installation complete!')
"

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Run tests: python -m pytest doc_nmt_mamba/tests/ -v"
echo "  2. Train: python doc_nmt_mamba/scripts/train.py"
echo ""
