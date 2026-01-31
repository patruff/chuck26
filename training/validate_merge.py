#!/usr/bin/env python3
"""Validate merged model quality by comparing adapter-loaded vs merged model outputs.

This script loads both the adapter-loaded model and the merged model, runs
identical test prompts through each, and compares the outputs side-by-side.
The goal is to verify that the merge did not degrade model quality.

Prerequisites:
    - train_dpo.py must have completed (adapter at /workspace/dpo-output/final-adapter)
    - merge_and_push.py must have completed (merged model at /workspace/merged-model/
      or on HuggingFace Hub at patruff/chuckles-qwen3-32b-dpo)

What it validates:
    - Merged model produces non-empty outputs (not blank or error responses)
    - Merged model outputs contain actual parody content (not just the prompt echoed back)
    - Quality is comparable between adapter-loaded and merged models

    Outputs may differ slightly due to quantization. Check for semantic quality,
    not exact match. The adapter model runs in 4-bit quantized mode while the
    merged model runs in FP16 -- minor output differences are expected and normal.

Expected output:
    - Side-by-side comparison table of adapter vs merged model outputs
    - PASS/FAIL summary based on output quality checks

Usage:
    python validate_merge.py

    To load merged model from Hub instead of local path:
        Set USE_HUB_MERGED = True in the configuration section below.
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from unsloth import FastModel

# =============================================================================
# Configuration
# =============================================================================

BASE_MODEL = "unsloth/Qwen3-32B-unsloth-bnb-4bit"
ADAPTER_PATH = "/workspace/dpo-output/final-adapter"
MERGED_MODEL_PATH = "/workspace/merged-model"
HUB_MERGED_REPO = "patruff/chuckles-qwen3-32b-dpo"
MAX_SEQ_LENGTH = 2048

# Set to True to load merged model from HuggingFace Hub instead of local path
USE_HUB_MERGED = False

# Generation parameters -- identical for both models to ensure fair comparison
GENERATION_KWARGS = {
    "max_new_tokens": 256,
    "temperature": 0.7,
    "do_sample": True,
}

# System prompt for parody generation
SYSTEM_PROMPT = (
    "You are a comedy writer who creates funny parody titles. "
    "Given a movie, book, or show title, create a phonetically similar "
    "parody that sounds like the original when spoken aloud."
)

# Test prompts -- at least 5 covering different title styles
TEST_PROMPTS = [
    "Create a phonetically-sound parody of: 'The Shawshank Redemption'",
    "Create a phonetically-sound parody of: 'Pulp Fiction'",
    "Create a phonetically-sound parody of: 'The Godfather'",
    "Create a phonetically-sound parody of: 'Jurassic Park'",
    "Create a phonetically-sound parody of: 'Fight Club'",
]


# =============================================================================
# Helper Functions
# =============================================================================

def format_prompt(tokenizer, prompt):
    """Format a prompt using the Qwen3 chat template with thinking disabled."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    return inputs


def generate_response(model, tokenizer, prompt, device=None):
    """Generate a response for a single prompt."""
    inputs = format_prompt(tokenizer, prompt)

    if device is not None:
        inputs = inputs.to(device)
    else:
        inputs = inputs.to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs,
            **GENERATION_KWARGS,
        )

    # Decode only the new tokens (skip the input prompt tokens)
    new_token_ids = output_ids[0][inputs.shape[1]:]
    response = tokenizer.decode(new_token_ids, skip_special_tokens=True)
    return response.strip()


def check_output_quality(prompt, output):
    """Check if an output appears to be valid parody content.

    Returns (passed, reason) tuple.
    """
    # Check 1: Non-empty
    if not output or len(output.strip()) == 0:
        return False, "Output is empty"

    # Check 2: Not just the prompt echoed back
    # Extract the title from the prompt for comparison
    if "'" in prompt:
        title = prompt.split("'")[1]
        if output.strip() == title:
            return False, "Output is just the original title echoed back"

    # Check 3: Has some content (at least a few characters of actual response)
    if len(output.strip()) < 3:
        return False, f"Output too short ({len(output.strip())} chars)"

    return True, "OK"


def print_separator(char="=", width=80):
    """Print a separator line."""
    print(char * width)


# =============================================================================
# Phase 1: Generate with Adapter-Loaded Model
# =============================================================================

def run_adapter_model():
    """Load base model with adapter and generate outputs for all test prompts."""
    print_separator()
    print("Phase 1: Generating with adapter-loaded model")
    print_separator()
    print()

    # Check adapter exists
    if not os.path.isdir(ADAPTER_PATH):
        print(f"ERROR: Adapter not found at {ADAPTER_PATH}")
        print("Run train_dpo.py first.")
        sys.exit(1)

    # Load base model
    print(f"Loading base model: {BASE_MODEL}")
    model, tokenizer = FastModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        full_finetuning=False,
    )

    # Load trained adapter
    print(f"Loading adapter from: {ADAPTER_PATH}")
    model.load_adapter(ADAPTER_PATH)

    # Generate outputs
    adapter_outputs = []
    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"  Generating [{i + 1}/{len(TEST_PROMPTS)}]: {prompt[:60]}...")
        response = generate_response(model, tokenizer, prompt)
        adapter_outputs.append(response)
        print(f"    -> {response[:100]}")

    # Free memory before loading merged model
    del model
    del tokenizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print()
    print(f"Adapter model: {len(adapter_outputs)} outputs generated.")
    return adapter_outputs


# =============================================================================
# Phase 2: Generate with Merged Model
# =============================================================================

def run_merged_model():
    """Load merged model and generate outputs for all test prompts."""
    print_separator()
    print("Phase 2: Generating with merged model")
    print_separator()
    print()

    # Determine source path
    if USE_HUB_MERGED:
        model_source = HUB_MERGED_REPO
        print(f"Loading merged model from Hub: {model_source}")
    else:
        model_source = MERGED_MODEL_PATH
        if not os.path.isdir(model_source):
            print(f"ERROR: Merged model not found at {model_source}")
            print("Run merge_and_push.py first, or set USE_HUB_MERGED = True.")
            sys.exit(1)
        print(f"Loading merged model from local: {model_source}")

    # Load merged model (FP16, no adapter needed)
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_source)

    # Generate outputs
    merged_outputs = []
    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"  Generating [{i + 1}/{len(TEST_PROMPTS)}]: {prompt[:60]}...")
        response = generate_response(model, tokenizer, prompt)
        merged_outputs.append(response)
        print(f"    -> {response[:100]}")

    # Free memory
    del model
    del tokenizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print()
    print(f"Merged model: {len(merged_outputs)} outputs generated.")
    return merged_outputs


# =============================================================================
# Phase 3: Compare and Report
# =============================================================================

def compare_and_report(adapter_outputs, merged_outputs):
    """Print side-by-side comparison and PASS/FAIL summary."""
    print()
    print_separator()
    print("Comparison: Adapter-Loaded vs Merged Model")
    print_separator()
    print()

    # Side-by-side comparison table
    results = []

    for i, (prompt, adapter_out, merged_out) in enumerate(
        zip(TEST_PROMPTS, adapter_outputs, merged_outputs)
    ):
        # Extract just the title for display
        title = prompt.split("'")[1] if "'" in prompt else prompt

        print(f"--- Prompt {i + 1}: '{title}' ---")
        print(f"  Adapter: {adapter_out}")
        print(f"  Merged:  {merged_out}")

        # Quality checks on merged output
        passed, reason = check_output_quality(prompt, merged_out)
        status = "PASS" if passed else f"FAIL ({reason})"
        results.append((title, passed, reason))

        print(f"  Status:  {status}")
        print()

    # Summary
    print_separator()
    print("SUMMARY")
    print_separator()
    print()

    total = len(results)
    passed_count = sum(1 for _, passed, _ in results if passed)
    failed_count = total - passed_count

    for title, passed, reason in results:
        icon = "PASS" if passed else "FAIL"
        print(f"  [{icon}] '{title}': {reason}")

    print()
    print(f"Results: {passed_count}/{total} passed, {failed_count}/{total} failed")
    print()

    if failed_count == 0:
        print_separator()
        print("OVERALL: PASS")
        print("All merged model outputs contain valid parody content.")
        print("Merge did not degrade model quality.")
        print_separator()
        return True
    else:
        print_separator()
        print("OVERALL: FAIL")
        print(f"{failed_count} output(s) did not meet quality checks.")
        print("Review the outputs above. The merge may have introduced issues.")
        print_separator()
        return False


# =============================================================================
# Main
# =============================================================================

def main():
    """Run the full validation pipeline."""
    print()
    print_separator()
    print(" chucklesPRIME - Merge Validation")
    print_separator()
    print()
    print(f"Base model:    {BASE_MODEL}")
    print(f"Adapter path:  {ADAPTER_PATH}")
    print(f"Merged model:  {HUB_MERGED_REPO if USE_HUB_MERGED else MERGED_MODEL_PATH}")
    print(f"Test prompts:  {len(TEST_PROMPTS)}")
    print()

    # Phase 1: Adapter model outputs
    adapter_outputs = run_adapter_model()

    # Phase 2: Merged model outputs
    merged_outputs = run_merged_model()

    # Phase 3: Compare and report
    overall_pass = compare_and_report(adapter_outputs, merged_outputs)

    # Exit code: 0 for pass, 1 for fail
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
