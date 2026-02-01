#!/bin/bash
# =============================================================================
# vLLM Inference Server Setup for chucklesPRIME
# =============================================================================
#
# Launches a vLLM server on a RunPod GPU pod to serve the fine-tuned
# Qwen3-32B model via an OpenAI-compatible API.
#
# Prerequisites:
#   - RunPod pod with A6000 48GB (recommended) or RTX 4090 24GB
#   - Network volume mounted at /workspace/ (200GB+ recommended)
#   - HF_TOKEN environment variable set (export HF_TOKEN="hf_...")
#
# Usage:
#   chmod +x setup_inference.sh
#   ./setup_inference.sh --test   # Small model (Qwen3-0.6B) to validate pipeline format
#   ./setup_inference.sh --bnb    # BitsAndBytes NF4 runtime quantization (default)
#   ./setup_inference.sh --awq    # Serve pre-quantized AWQ model (faster)
#   ./setup_inference.sh --help   # Show usage
#
# The server exposes an OpenAI-compatible API on port 8000.
# The existing chucklesPRIME CLI can connect with zero code changes --
# just update settings.json with the pod's API base URL.
# =============================================================================
set -e

# =============================================================================
# Configuration
# =============================================================================

# FP16 merged model on Hub (produced by merge_and_push.py in Phase 5)
MODEL_ID="patruff/chuckles-qwen3-32b-dpo"

# Pre-quantized AWQ model on Hub (produced by quantize_awq.py)
AWQ_MODEL_ID="patruff/chuckles-qwen3-32b-dpo-awq"

# Small test model -- validates pipeline format without needing expensive GPU.
# Same Qwen3 architecture family, so chat template and API behavior are identical.
TEST_MODEL_ID="Qwen/Qwen3-0.6B"

# Served model name -- MUST match settings.json model_name field.
# Without this, vLLM uses the full HF repo ID and requests fail with 404.
SERVED_NAME="chuckles-qwen3-32b-dpo"

# Server configuration
PORT=8000

# CRITICAL: Default Qwen3-32B context is 40960 tokens, which causes OOM on
# all consumer GPUs. Parodies need < 8K tokens, so 8192 is more than enough.
MAX_MODEL_LEN=8192

# Use 90% of VRAM for model weights + KV cache
GPU_MEM_UTIL=0.90

# =============================================================================
# Parse Command-Line Arguments
# =============================================================================

QUANT_MODE="bnb"  # default

case "${1:-}" in
    --test)
        QUANT_MODE="test"
        ;;
    --bnb)
        QUANT_MODE="bnb"
        ;;
    --awq)
        QUANT_MODE="awq"
        ;;
    --help|-h)
        echo "Usage: $0 [--test | --bnb | --awq | --help]"
        echo ""
        echo "Options:"
        echo "  --test  Small model (Qwen3-0.6B) to validate the pipeline format."
        echo "          Fits on any GPU (~1.2GB). Tests API format, settings.json"
        echo "          integration, and validate_inference.py without burning GPU"
        echo "          hours on the full 32B model. Use this first!"
        echo ""
        echo "  --bnb   BitsAndBytes NF4 runtime quantization (default)"
        echo "          Simpler setup, no pre-quantization needed."
        echo "          ~168 tok/s throughput. Needs ~65GB+ system RAM for loading."
        echo ""
        echo "  --awq   Serve pre-quantized AWQ 4-bit model (Marlin kernel)"
        echo "          Faster inference (~712 tok/s). Requires running"
        echo "          quantize_awq.py first to create the AWQ model."
        echo ""
        echo "  --help  Show this usage information"
        exit 0
        ;;
    "")
        QUANT_MODE="bnb"
        ;;
    *)
        echo "ERROR: Unknown option '$1'"
        echo "Usage: $0 [--test | --bnb | --awq | --help]"
        exit 1
        ;;
esac

# In test mode, override served name to match the test model
if [ "$QUANT_MODE" = "test" ]; then
    SERVED_NAME="$TEST_MODEL_ID"
fi

echo "============================================================"
echo " chucklesPRIME - vLLM Inference Server"
echo "============================================================"
echo ""
echo " Quantization mode: $QUANT_MODE"
if [ "$QUANT_MODE" = "test" ]; then
    echo " Model:             $TEST_MODEL_ID (small test model)"
else
    echo " Model:             $MODEL_ID"
fi
echo " Served as:         $SERVED_NAME"
echo " Port:              $PORT"
echo " Max context:       $MAX_MODEL_LEN tokens"
echo ""
echo "============================================================"

# =============================================================================
# Step 1: Verify HF_TOKEN is set
# =============================================================================
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

echo "[1/4] HF_TOKEN is set."

# =============================================================================
# Step 2: Pre-flight checks
# =============================================================================
echo ""
echo "[2/4] Running pre-flight checks..."

# Check GPU is available
if ! nvidia-smi > /dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found or no GPU detected."
    echo "This script requires a GPU pod (A6000 48GB recommended)."
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "  GPU: $GPU_NAME (${VRAM_MB}MB VRAM)"

# For BitsAndBytes mode, check system RAM (needs ~65GB+ for loading)
if [ "$QUANT_MODE" = "bnb" ]; then
    SYSTEM_RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
    echo "  System RAM: ${SYSTEM_RAM_GB}GB"
    if [ "$SYSTEM_RAM_GB" -lt 65 ]; then
        echo ""
        echo "WARNING: System RAM is ${SYSTEM_RAM_GB}GB, BitsAndBytes mode needs ~65GB+"
        echo "         for loading the FP16 model into CPU memory before quantizing."
        echo "         The server may fail to start or be very slow."
        echo "         Consider using --awq mode instead."
        echo ""
    fi
fi

echo "  Pre-flight checks passed."

# =============================================================================
# Step 3: Install vLLM
# =============================================================================
echo ""
echo "[3/4] Installing vLLM..."

if python3 -c "import vllm; print(f'vLLM {vllm.__version__} already installed')" 2>/dev/null; then
    echo "  vLLM is already installed, skipping."
else
    pip install "vllm>=0.15.0"
    echo "  vLLM installed successfully."
fi

# =============================================================================
# Step 4: Launch vLLM server
# =============================================================================
echo ""
echo "[4/4] Launching vLLM server..."
echo ""

# Select model and quantization flags based on mode
if [ "$QUANT_MODE" = "test" ]; then
    LAUNCH_MODEL="$TEST_MODEL_ID"
    QUANT_FLAGS=""
    echo "  Mode: TEST (small Qwen3-0.6B -- pipeline format validation)"
    echo "  Model: $LAUNCH_MODEL"
    echo ""
    echo "  This validates the full pipeline format:"
    echo "    - vLLM startup and API exposure"
    echo "    - OpenAI-compatible chat completions"
    echo "    - validate_inference.py connectivity"
    echo "    - settings.json integration with CLI"
    echo ""
    echo "  The model is tiny (~1.2GB). Output quality doesn't matter --"
    echo "  you're testing that the FORMAT works, not the CONTENT."
elif [ "$QUANT_MODE" = "bnb" ]; then
    LAUNCH_MODEL="$MODEL_ID"
    QUANT_FLAGS="--quantization bitsandbytes"
    echo "  Mode: BitsAndBytes NF4 (runtime quantization)"
    echo "  Model: $LAUNCH_MODEL"
elif [ "$QUANT_MODE" = "awq" ]; then
    LAUNCH_MODEL="$AWQ_MODEL_ID"
    QUANT_FLAGS=""
    echo "  Mode: AWQ 4-bit (pre-quantized, Marlin kernel)"
    echo "  Model: $LAUNCH_MODEL"
fi

echo ""
echo "============================================================"
echo " Server starting on port $PORT..."
echo "============================================================"
echo ""
echo " Once running, test with:"
echo "   curl http://localhost:$PORT/v1/models"
echo ""
if [ "$QUANT_MODE" = "test" ]; then
    echo " This is a FORMAT TEST -- validate the pipeline with:"
    echo "   python validate_inference.py --finetuned-url http://localhost:$PORT/v1 --finetuned-model $SERVED_NAME"
    echo ""
    echo " Output quality will be low (0.6B model). You're verifying:"
    echo "   - API responds to chat completions"
    echo "   - validate_inference.py scoring works"
    echo "   - settings.json integration format is correct"
else
    echo " Update your settings.json with:"
    echo "   \"api_base_url\": \"http://<POD_IP>:$PORT/v1\""
    echo "   \"model_name\": \"$SERVED_NAME\""
    echo ""
    echo " Set API key env var (vLLM doesn't require one, but the CLI expects it):"
    echo "   export VLLM_API_KEY='not-needed'"
fi
echo ""
echo "============================================================"
echo ""

# Launch vLLM with all critical flags:
#   --served-model-name: MUST match settings.json model_name (otherwise 404)
#   --max-model-len 8192: Prevents OOM (default 40960 is too large)
#   --default-chat-template-kwargs: Disables Qwen3 thinking mode (no <think> blocks)
#   --gpu-memory-utilization: Use 90% of VRAM
#   --host 0.0.0.0: Bind all interfaces (required for RunPod port forwarding)
#   --dtype half: Explicit FP16 (matches merged model dtype)
# shellcheck disable=SC2086
vllm serve "$LAUNCH_MODEL" \
    --served-model-name "$SERVED_NAME" \
    --max-model-len "$MAX_MODEL_LEN" \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --dtype half \
    $QUANT_FLAGS
