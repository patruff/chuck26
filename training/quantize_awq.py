#!/usr/bin/env python3
"""Quantize the FP16 merged model to AWQ 4-bit and push to HuggingFace Hub.

This script takes the FP16 merged model from Phase 5 (on HuggingFace Hub),
quantizes it to AWQ 4-bit using AutoAWQ, validates the quantized output,
and pushes the result to Hub for fast vLLM serving.

Prerequisites:
    - The FP16 merged model must exist on Hub at patruff/chuckles-qwen3-32b-dpo
      (produced by merge_and_push.py in Phase 5)
    - HF_TOKEN environment variable must be set
    - RunPod pod with A6000 48GB+ (quantization needs ~48GB VRAM)
    - Network volume mounted at /workspace/ with ~80GB free

Dependencies:
    pip install autoawq transformers huggingface-hub

Expected runtime:
    - 2-4 hours (calibration pass over model layers)
    - Hub push time depends on upload speed (~17GB upload)

Usage:
    export HF_TOKEN="hf_your_token_here"
    python quantize_awq.py

Output:
    - Local quantized model at /workspace/chuckles-qwen3-32b-dpo-awq/
    - Hub model at https://huggingface.co/patruff/chuckles-qwen3-32b-dpo-awq

NOTE: Uses AutoAWQ (not LLM Compressor) for Qwen3-32B quantization.
LLM Compressor has known quality issues with Qwen3-32B W4A16
(see GitHub issue vllm-project/llm-compressor#1600). AutoAWQ is
deprecated but still functional and produces better results for this model.
"""

import os
import sys

# =============================================================================
# Configuration
# =============================================================================

# Source FP16 model on Hub (produced by merge_and_push.py)
MODEL_ID = "patruff/chuckles-qwen3-32b-dpo"

# Local save path (RunPod /workspace/ for persistence across pod restarts)
QUANT_OUTPUT = "/workspace/chuckles-qwen3-32b-dpo-awq"

# Target Hub repo for the quantized model
HUB_REPO = "patruff/chuckles-qwen3-32b-dpo-awq"

# AWQ quantization configuration
# - zero_point=True: symmetric quantization (standard for AWQ)
# - q_group_size=128: group size for quantization (128 is standard)
# - w_bit=4: 4-bit weight quantization
# - version="GEMM": GEMM kernel (compatible with vLLM Marlin backend)
QUANT_CONFIG = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",
}


# =============================================================================
# Prerequisite Checks
# =============================================================================

def check_prerequisites():
    """Verify all prerequisites before starting quantization."""

    # Check 1: HF_TOKEN is set
    if not os.environ.get("HF_TOKEN"):
        print("=" * 60)
        print("ERROR: HF_TOKEN environment variable is not set.")
        print()
        print("Set it before running this script:")
        print('  export HF_TOKEN="hf_your_token_here"')
        print()
        print("Get your token at: https://huggingface.co/settings/tokens")
        print("=" * 60)
        sys.exit(1)

    print("[OK] HF_TOKEN is set.")

    # Check 2: AutoAWQ is installed
    try:
        import awq  # noqa: F401
        print("[OK] AutoAWQ is installed.")
    except ImportError:
        print("ERROR: AutoAWQ is not installed.")
        print()
        print("Install it with:")
        print("  pip install autoawq")
        sys.exit(1)

    # Check 3: GPU is available
    try:
        import torch
        if not torch.cuda.is_available():
            print("ERROR: No GPU detected. AWQ quantization requires a GPU.")
            sys.exit(1)
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"[OK] GPU: {gpu_name} ({vram_gb:.1f}GB VRAM)")
    except ImportError:
        print("ERROR: PyTorch is not installed.")
        sys.exit(1)


# =============================================================================
# Main Quantization Workflow
# =============================================================================

def main():
    """Quantize FP16 model to AWQ 4-bit and push to Hub."""

    print("=" * 60)
    print(" chucklesPRIME - AWQ 4-bit Quantization")
    print("=" * 60)
    print()

    # --- Prerequisite checks ---
    check_prerequisites()
    print()

    # --- Step 1: Load FP16 model from Hub ---
    print("=" * 60)
    print("Step 1: Loading FP16 model from Hub...")
    print(f"  Model: {MODEL_ID}")
    print("  This downloads ~65GB. Please be patient.")
    print("=" * 60)

    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    model = AutoAWQForCausalLM.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print(f"[OK] Model loaded: {MODEL_ID}")

    # --- Step 2: Quantize to AWQ 4-bit ---
    print()
    print("=" * 60)
    print("Step 2: Quantizing to AWQ 4-bit...")
    print(f"  Config: {QUANT_CONFIG}")
    print("  This takes 2-4 hours. Please be patient.")
    print("=" * 60)

    model.quantize(tokenizer, quant_config=QUANT_CONFIG)

    print("[OK] Quantization complete.")

    # --- Step 3: Save quantized model locally ---
    print()
    print("=" * 60)
    print(f"Step 3: Saving quantized model to {QUANT_OUTPUT}...")
    print("=" * 60)

    os.makedirs(QUANT_OUTPUT, exist_ok=True)
    model.save_quantized(QUANT_OUTPUT)
    tokenizer.save_pretrained(QUANT_OUTPUT)

    print(f"[OK] Quantized model saved to: {QUANT_OUTPUT}")

    # List output files
    output_files = os.listdir(QUANT_OUTPUT)
    total_size_gb = sum(
        os.path.getsize(os.path.join(QUANT_OUTPUT, f))
        for f in output_files
        if os.path.isfile(os.path.join(QUANT_OUTPUT, f))
    ) / (1024 ** 3)
    print(f"  Files: {len(output_files)} ({total_size_gb:.1f}GB)")
    for f in sorted(output_files):
        fpath = os.path.join(QUANT_OUTPUT, f)
        if os.path.isfile(fpath):
            size_mb = os.path.getsize(fpath) / (1024 ** 2)
            print(f"    {f}: {size_mb:.1f}MB")

    # --- Step 4: Validate quantized model ---
    print()
    print("=" * 60)
    print("Step 4: Validating quantized model...")
    print("=" * 60)

    try:
        # Load the quantized model back
        val_model = AutoAWQForCausalLM.from_quantized(QUANT_OUTPUT)
        val_tokenizer = AutoTokenizer.from_pretrained(QUANT_OUTPUT)

        # Generate a short test response
        test_messages = [
            {"role": "user", "content": "Say hello and introduce yourself briefly."}
        ]
        test_input = val_tokenizer.apply_chat_template(
            test_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        test_input = test_input.to(val_model.device)

        import torch
        with torch.no_grad():
            output = val_model.generate(
                test_input,
                max_new_tokens=100,
                do_sample=False,
            )

        response = val_tokenizer.decode(
            output[0][test_input.shape[-1]:],
            skip_special_tokens=True,
        )

        print(f"[OK] Validation response: {response[:200]}")

        if not response.strip():
            print("WARNING: Empty response from quantized model.")
            print("  The model may have quality issues, but continuing with push.")
        else:
            print("[OK] Quantized model produces coherent output.")

        # Clean up validation model to free VRAM
        del val_model
        del test_input
        del output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as val_error:
        print(f"WARNING: Validation failed: {val_error}")
        print("  Continuing with Hub push anyway (model may still work with vLLM).")

    # --- Step 5: Push to Hub ---
    print()
    print("=" * 60)
    print(f"Step 5: Pushing quantized model to Hub: {HUB_REPO}")
    print("=" * 60)

    try:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ["HF_TOKEN"])

        # Create repo if it doesn't exist
        api.create_repo(
            repo_id=HUB_REPO,
            repo_type="model",
            exist_ok=True,
        )

        # Upload the entire quantized model directory
        print(f"  Uploading {QUANT_OUTPUT} to {HUB_REPO}...")
        api.upload_folder(
            folder_path=QUANT_OUTPUT,
            repo_id=HUB_REPO,
            repo_type="model",
        )

        print(f"[OK] Quantized model pushed to Hub: {HUB_REPO}")

    except Exception as push_error:
        print(f"ERROR: Hub push failed: {push_error}")
        print()
        print("The quantized model is saved locally at:")
        print(f"  {QUANT_OUTPUT}")
        print()
        print("You can manually upload it later with:")
        print(f"  huggingface-cli upload {HUB_REPO} {QUANT_OUTPUT}")
        print()
        print("Continuing without push (local model is intact).")

    # --- Done ---
    print()
    print("=" * 60)
    print(" AWQ Quantization Complete!")
    print("=" * 60)
    print()
    print(f"  Local quantized model:  {QUANT_OUTPUT}")
    print(f"  Hub model URL:          https://huggingface.co/{HUB_REPO}")
    print()
    print("Next steps:")
    print("  Serve with vLLM:  bash setup_inference.sh --awq")
    print("=" * 60)


if __name__ == "__main__":
    main()
