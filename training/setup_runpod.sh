#!/bin/bash
# =============================================================================
# RunPod Environment Setup for chucklesPRIME DPO Training
# =============================================================================
#
# Sets up a fresh RunPod pod for DPO fine-tuning of Qwen3-32B.
#
# Prerequisites:
#   - RunPod pod with A6000 48GB (or A100 80GB)
#   - Network volume mounted at /workspace/ (200GB+ recommended)
#   - HF_TOKEN environment variable set (export HF_TOKEN="hf_...")
#
# Usage:
#   chmod +x setup_runpod.sh && ./setup_runpod.sh
#
# IMPORTANT: Network volume must be at least 200GB to accommodate:
#   - Training checkpoints (~5-10GB)
#   - LoRA adapter (~100-300MB)
#   - Merged 16-bit model (~65GB) during export step
#   - FP16 base model download (~65GB) needed for merge
#
# All output paths MUST use /workspace/ to survive pod restarts.
# Anything outside /workspace/ is ephemeral container storage.
# =============================================================================
set -e

echo "============================================================"
echo " chucklesPRIME - RunPod DPO Training Setup"
echo "============================================================"
echo ""
echo " This script installs all dependencies for DPO fine-tuning"
echo " of Qwen3-32B using Unsloth + TRL on RunPod."
echo ""
echo "============================================================"

# ---------------------------------------------------------------------------
# Step 1: Verify HF_TOKEN is set (needed for Hub access)
# ---------------------------------------------------------------------------
if [ -z "$HF_TOKEN" ]; then
    echo ""
    echo "ERROR: HF_TOKEN environment variable is not set."
    echo ""
    echo "Set it before running this script:"
    echo "  export HF_TOKEN=\"hf_your_token_here\""
    echo ""
    echo "Get your token at: https://huggingface.co/settings/tokens"
    exit 1
fi

echo "[1/5] HF_TOKEN is set."

# ---------------------------------------------------------------------------
# Step 2: Install Unsloth FIRST
# ---------------------------------------------------------------------------
# CRITICAL: Unsloth must be installed before TRL.
# Unsloth monkey-patches TRL and Transformers internals for 2x speedup and
# 70% VRAM reduction. It pins specific versions of torch, transformers, peft,
# and bitsandbytes that it has been tested with. Installing TRL first would
# pull incompatible versions, causing method signature mismatches and crashes
# (see Pitfall 3 in research: Unsloth/TRL version mismatch).
# ---------------------------------------------------------------------------
echo ""
echo "[2/5] Installing Unsloth (this controls all dependency versions)..."
pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo

# ---------------------------------------------------------------------------
# Step 3: Install TRL (DPOTrainer)
# ---------------------------------------------------------------------------
# Installed AFTER Unsloth so Unsloth's version pins take precedence.
# TRL >= 0.27.1 required for current DPOConfig/DPOTrainer API.
# ---------------------------------------------------------------------------
echo ""
echo "[3/5] Installing TRL..."
pip install "trl>=0.27.1"

# ---------------------------------------------------------------------------
# Step 4: Install optional dependencies
# ---------------------------------------------------------------------------
# wandb is optional -- enables training metrics dashboards (loss curves,
# reward margins, gradient norms). Set report_to="wandb" in DPOConfig
# if you want to use it.
# ---------------------------------------------------------------------------
echo ""
echo "[4/5] Installing optional dependencies (wandb)..."
pip install "wandb>=0.19.0" || echo "WARNING: wandb install failed (optional, continuing)"

# ---------------------------------------------------------------------------
# Step 5: Login to HuggingFace Hub
# ---------------------------------------------------------------------------
echo ""
echo "[5/5] Logging in to HuggingFace Hub..."
huggingface-cli login --token "$HF_TOKEN"

# ---------------------------------------------------------------------------
# Verify installation
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " Installation Verification"
echo "============================================================"
python3 -c "
import unsloth; print(f'  unsloth:        {unsloth.__version__}')
import trl; print(f'  trl:            {trl.__version__}')
import transformers; print(f'  transformers:   {transformers.__version__}')
import torch; print(f'  torch:          {torch.__version__}')
import peft; print(f'  peft:           {peft.__version__}')
import bitsandbytes; print(f'  bitsandbytes:   {bitsandbytes.__version__}')
print()
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:            {torch.cuda.get_device_name(0)}')
    print(f'  VRAM:           {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

echo ""
echo "============================================================"
echo " Setup Complete!"
echo "============================================================"
echo ""
echo " Next steps:"
echo "   1. Copy train_dpo.py to this directory"
echo "   2. Run: python train_dpo.py"
echo ""
echo " All outputs will be saved to /workspace/dpo-output/"
echo " Adapter will be pushed to HuggingFace Hub automatically."
echo "============================================================"
