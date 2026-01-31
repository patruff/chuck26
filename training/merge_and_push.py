#!/usr/bin/env python3
"""Merge LoRA adapter into FP16 base model and push to HuggingFace Hub.

This script takes the trained LoRA adapter from DPO training (produced by
train_dpo.py) and merges it into the full FP16 base model weights. The merged
model is saved locally first (safety net), then pushed to HuggingFace Hub.

Prerequisites:
    - train_dpo.py must have completed successfully, producing an adapter at
      /workspace/dpo-output/final-adapter
    - HF_TOKEN environment variable must be set
    - Network volume mounted at /workspace/ with ~130GB free space

Disk space requirements:
    - ~65GB for FP16 base model download (Unsloth re-downloads FP16 weights
      during merge -- this is a known behavior, see Unsloth issue #3633)
    - ~65GB for merged model output (safetensors files)
    - Total: ~130GB minimum free on /workspace/

Expected runtime:
    - 30-60 minutes (mostly FP16 base model download + merge computation)
    - Hub push time depends on upload speed (~65GB upload)

Usage:
    export HF_TOKEN="hf_your_token_here"
    python merge_and_push.py

Output:
    - Local merged model at /workspace/merged-model/
    - Hub model at https://huggingface.co/patruff/chuckles-qwen3-32b-dpo
"""

import os
import sys
import shutil
import glob

from unsloth import FastModel

# =============================================================================
# Configuration
# =============================================================================

BASE_MODEL = "unsloth/Qwen3-32B-unsloth-bnb-4bit"
ADAPTER_PATH = "/workspace/dpo-output/final-adapter"
MERGED_OUTPUT_DIR = "/workspace/merged-model"
HUB_MERGED_REPO = "patruff/chuckles-qwen3-32b-dpo"
MAX_SEQ_LENGTH = 2048

# Minimum recommended free disk space in GB for the merge operation
MIN_DISK_SPACE_GB = 150


# =============================================================================
# Prerequisite Checks
# =============================================================================

def check_prerequisites():
    """Verify all prerequisites before starting the merge."""

    # Check 1: Adapter path exists (means train_dpo.py has been run)
    if not os.path.isdir(ADAPTER_PATH):
        print("=" * 60)
        print("ERROR: Adapter not found at:")
        print(f"  {ADAPTER_PATH}")
        print()
        print("This means train_dpo.py has not been run yet, or the adapter")
        print("was saved to a different location.")
        print()
        print("Run train_dpo.py first to produce the LoRA adapter.")
        print("=" * 60)
        sys.exit(1)

    print(f"[OK] Adapter found at {ADAPTER_PATH}")

    # List adapter files for verification
    adapter_files = os.listdir(ADAPTER_PATH)
    print(f"     Adapter files: {adapter_files}")

    # Check 2: HF_TOKEN is set (needed for Hub push)
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

    # Check 3: Disk space on /workspace/
    try:
        disk_usage = shutil.disk_usage("/workspace/")
        free_gb = disk_usage.free / (1024 ** 3)
        total_gb = disk_usage.total / (1024 ** 3)
        print(f"[OK] Disk space: {free_gb:.1f} GB free / {total_gb:.1f} GB total on /workspace/")

        if free_gb < MIN_DISK_SPACE_GB:
            print()
            print("=" * 60)
            print(f"WARNING: Less than {MIN_DISK_SPACE_GB}GB free on /workspace/")
            print(f"         Merge requires ~130GB (65GB FP16 download + 65GB output)")
            print(f"         Available: {free_gb:.1f}GB")
            print()
            print("The merge may fail due to insufficient disk space.")
            print("Consider clearing unused checkpoints or expanding the volume.")
            print("=" * 60)
            print()
            # Warning only, not a hard stop -- user may know better
    except OSError:
        print("[WARN] Could not check disk space on /workspace/")


# =============================================================================
# Main Merge Workflow
# =============================================================================

def main():
    """Merge LoRA adapter into FP16 base model and push to Hub."""

    print("=" * 60)
    print(" chucklesPRIME - LoRA Merge & Hub Push")
    print("=" * 60)
    print()

    # --- Prerequisite checks ---
    check_prerequisites()
    print()

    # --- Step 1: Load base model ---
    print("=" * 60)
    print("Step 1: Loading base model...")
    print("=" * 60)

    model, tokenizer = FastModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        full_finetuning=False,
    )

    print(f"Base model loaded: {BASE_MODEL}")

    # --- Step 2: Load trained adapter ---
    print()
    print("=" * 60)
    print("Step 2: Loading trained LoRA adapter...")
    print("=" * 60)

    model.load_adapter(ADAPTER_PATH)

    print(f"Adapter loaded from: {ADAPTER_PATH}")

    # --- Step 3: Merge to 16-bit and save locally ---
    # NOTE: This step will download the FP16 base model (~65GB) behind the scenes.
    # This is a known Unsloth behavior (issue #3633) -- the merged_16bit save method
    # needs the original FP16 weights to merge LoRA deltas into. The download goes
    # into a .cache/ folder inside MERGED_OUTPUT_DIR. This is expected and normal,
    # but takes 15-30 minutes depending on network speed.
    print()
    print("=" * 60)
    print("Step 3: Merging to 16-bit (saving locally first)...")
    print("  NOTE: This will download the FP16 base model (~65GB) behind the scenes.")
    print("  This is expected behavior (Unsloth issue #3633). Please be patient.")
    print("=" * 60)

    # CRITICAL: Always use merged_16bit. NEVER use a 4-bit merge method.
    # A 4-bit merge degrades quality due to compounding quantization errors.
    # LoRA weights were trained to compensate for 4-bit representation;
    # merging back into 4-bit double-quantizes and destroys fine-tuning gains.
    model.save_pretrained_merged(
        MERGED_OUTPUT_DIR,
        tokenizer,
        save_method="merged_16bit",
    )

    print(f"Merged model saved locally to: {MERGED_OUTPUT_DIR}")

    # --- Step 4: Verify merged model files exist on disk ---
    print()
    print("=" * 60)
    print("Step 4: Verifying merged model files...")
    print("=" * 60)

    safetensor_files = glob.glob(os.path.join(MERGED_OUTPUT_DIR, "*.safetensors"))
    config_file = os.path.join(MERGED_OUTPUT_DIR, "config.json")

    if not safetensor_files:
        print("ERROR: No .safetensors files found in merged output directory!")
        print(f"  Directory: {MERGED_OUTPUT_DIR}")
        print(f"  Contents: {os.listdir(MERGED_OUTPUT_DIR)}")
        print("The merge may have failed. Check the output above for errors.")
        sys.exit(1)

    if not os.path.isfile(config_file):
        print("WARNING: config.json not found in merged output directory.")
        print("  The model may not load correctly without it.")

    total_size_gb = sum(os.path.getsize(f) for f in safetensor_files) / (1024 ** 3)
    print(f"[OK] Found {len(safetensor_files)} safetensor file(s) ({total_size_gb:.1f} GB)")
    for f in sorted(safetensor_files):
        size_gb = os.path.getsize(f) / (1024 ** 3)
        print(f"     {os.path.basename(f)}: {size_gb:.2f} GB")

    # --- Step 5: Push merged model to Hub ---
    print()
    print("=" * 60)
    print(f"Step 5: Pushing merged model to Hub: {HUB_MERGED_REPO}")
    print("=" * 60)

    push_succeeded = False

    try:
        model.push_to_hub_merged(
            HUB_MERGED_REPO,
            tokenizer,
            save_method="merged_16bit",
            token=os.environ["HF_TOKEN"],
        )
        push_succeeded = True
        print(f"[OK] Model pushed to Hub via push_to_hub_merged.")
    except Exception as e:
        print(f"WARNING: push_to_hub_merged failed: {e}")
        print("Falling back to manual upload via huggingface_hub API...")

    # --- Step 5b: Fallback upload if push_to_hub_merged failed ---
    # Known bug #3146: push_to_hub_merged sometimes only pushes README,
    # not the actual model weights. The fallback uploads the entire
    # local merged model directory using HfApi.upload_folder().
    if not push_succeeded:
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=os.environ["HF_TOKEN"])

            print(f"Uploading {MERGED_OUTPUT_DIR} to {HUB_MERGED_REPO}...")
            api.create_repo(
                repo_id=HUB_MERGED_REPO,
                repo_type="model",
                exist_ok=True,
            )
            api.upload_folder(
                folder_path=MERGED_OUTPUT_DIR,
                repo_id=HUB_MERGED_REPO,
                repo_type="model",
            )
            push_succeeded = True
            print(f"[OK] Model uploaded to Hub via fallback (upload_folder).")
        except Exception as fallback_error:
            print(f"ERROR: Fallback upload also failed: {fallback_error}")
            print()
            print("The merged model is saved locally at:")
            print(f"  {MERGED_OUTPUT_DIR}")
            print()
            print("You can manually upload it later with:")
            print(f'  huggingface-cli upload {HUB_MERGED_REPO} {MERGED_OUTPUT_DIR}')
            sys.exit(1)

    # --- Done ---
    print()
    print("=" * 60)
    print(" Merge & Push Complete!")
    print("=" * 60)
    print()
    print(f"  Local merged model:  {MERGED_OUTPUT_DIR}")
    print(f"  Hub model URL:       https://huggingface.co/{HUB_MERGED_REPO}")
    print()
    print("Next steps:")
    print("  1. Run validate_merge.py to verify merged model quality")
    print("  2. Deploy with vLLM for inference (Phase 6)")
    print("=" * 60)


if __name__ == "__main__":
    main()
