#!/usr/bin/env python3
"""Validate inference quality by comparing fine-tuned vs base model outputs.

This script connects to one or two vLLM servers via the OpenAI-compatible API
and compares parody generation quality between the fine-tuned model and
(optionally) the base Qwen3-32B model on identical prompts.

Prerequisites:
    - vLLM server running with the fine-tuned model (setup_inference.sh)
    - Optionally: a second vLLM server running the base model for comparison

What it validates:
    - Fine-tuned model produces non-empty, parody-like outputs
    - Quality metrics: word count ratio (structure preservation),
      character similarity (phonetic overlap), and non-empty check
    - If base model provided: side-by-side comparison with scoring

Expected output:
    - Per-prompt quality scores and generated parodies
    - Summary table with average scores per model
    - PASS/FAIL verdict based on output quality

Usage:
    # Test fine-tuned model only (default):
    python validate_inference.py --finetuned-url http://localhost:8000/v1

    # Compare fine-tuned vs base model:
    python validate_inference.py \\
        --finetuned-url http://localhost:8000/v1 \\
        --base-url http://localhost:8001/v1

    # Print settings.json snippet for CLI integration:
    python validate_inference.py --print-settings
    python validate_inference.py --print-settings --finetuned-url http://1.2.3.4:8000/v1

Requires: pip install openai
"""

import argparse
import json
import sys
from difflib import SequenceMatcher
from urllib.parse import urlparse

# =============================================================================
# Configuration
# =============================================================================

# Default vLLM endpoints
FINETUNED_URL = "http://localhost:8000/v1"
BASE_URL = ""  # Empty = skip base model comparison

# Model names -- must match --served-model-name used when launching vLLM
FINETUNED_MODEL = "chuckles-qwen3-32b-dpo"
BASE_MODEL = "Qwen/Qwen3-32B-AWQ"

# vLLM does not require authentication, but the OpenAI client needs a value
API_KEY = "not-needed"

# Generation parameters -- identical for both models to ensure fair comparison
TEMPERATURE = 0.7
TOP_P = 0.8
MAX_TOKENS = 512

# System prompt for parody generation -- matches the style used in training
SYSTEM_PROMPT = (
    "You are a comedy writer who creates funny parody titles. "
    "Given a movie, book, or show title, create a phonetically similar "
    "parody that sounds like the original when spoken aloud."
)

# Prompt template -- wraps each title into a full generation prompt
PROMPT_TEMPLATE = "Create a phonetically-sound parody of: '{title}'"

# Test prompts -- 12 movie titles covering diverse phonetic challenges
# Mix of short, medium, and long titles for comprehensive testing
TEST_TITLES = [
    # Short titles (1-2 words) -- tests concise phonetic matching
    "Jaws",
    "Frozen",
    "Aliens",
    # Medium titles (2-3 words) -- tests multi-word phonetic flow
    "Pulp Fiction",
    "Fight Club",
    "Die Hard",
    "Top Gun",
    # Long titles (3+ words) -- tests extended phonetic chains
    "The Shawshank Redemption",
    "The Silence of the Lambs",
    "Jurassic Park",
    "Eternal Sunshine of the Spotless Mind",
    "No Country for Old Men",
]


# =============================================================================
# OpenAI Client Functions
# =============================================================================

def create_client(base_url):
    """Create an OpenAI client pointing at a vLLM server."""
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed.")
        print("Install with: pip install openai")
        sys.exit(1)

    return OpenAI(api_key=API_KEY, base_url=base_url)


def query_model(client, model_name, title):
    """Send a parody generation request and return the response text.

    Args:
        client: OpenAI client connected to vLLM.
        model_name: The served model name (must match --served-model-name).
        title: Movie title to generate a parody for.

    Returns:
        The model's response text, or "[ERROR: ...]" on failure.
    """
    prompt = PROMPT_TEMPLATE.format(title=title)
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
        )
        text = response.choices[0].message.content
        return text.strip() if text else "[ERROR: empty response]"
    except Exception as e:
        return f"[ERROR: {e}]"


# =============================================================================
# Scoring Functions
# =============================================================================

def score_parody(original_title, parody_text):
    """Compute simple quality metrics for a parody output.

    Metrics (all standalone -- no external phonetic tools needed):
        - word_count_ratio: len(parody) / len(original) -- closer to 1.0 means
          the parody preserves the original's word structure.
        - char_similarity: SequenceMatcher ratio between original and parody --
          measures overall character-level overlap (proxy for phonetic similarity).
        - non_empty: 1.0 if parody has real content, 0.0 if empty/error.

    Args:
        original_title: The input movie title.
        parody_text: The model's generated parody text.

    Returns:
        Dict with the three metric scores.
    """
    # Check for errors or empty output
    if not parody_text or parody_text.startswith("[ERROR"):
        return {
            "word_count_ratio": 0.0,
            "char_similarity": 0.0,
            "non_empty": 0.0,
        }

    # Extract just the parody title if the model gave a longer explanation.
    # Use the first line that isn't empty as the parody title.
    lines = [ln.strip() for ln in parody_text.strip().split("\n") if ln.strip()]
    parody_title = lines[0] if lines else parody_text.strip()

    # Remove common prefixes the model might add
    for prefix in ["Parody:", "Parody Title:", "Answer:", "Title:"]:
        if parody_title.lower().startswith(prefix.lower()):
            parody_title = parody_title[len(prefix):].strip()

    # Strip surrounding quotes if present
    if len(parody_title) >= 2 and parody_title[0] in ('"', "'") and parody_title[-1] in ('"', "'"):
        parody_title = parody_title[1:-1]

    orig_words = original_title.split()
    parody_words = parody_title.split()

    # Metric 1: Word count ratio (structure preservation)
    if len(orig_words) > 0:
        ratio = len(parody_words) / len(orig_words)
        # Cap at 2.0 to avoid inflated scores from verbose outputs
        word_count_ratio = min(ratio, 2.0) / 2.0 if ratio > 1.0 else ratio
    else:
        word_count_ratio = 0.0

    # Metric 2: Character similarity (phonetic overlap proxy)
    char_similarity = SequenceMatcher(
        None, original_title.lower(), parody_title.lower()
    ).ratio()

    # Metric 3: Non-empty check
    non_empty = 1.0 if len(parody_title) > 0 else 0.0

    return {
        "word_count_ratio": round(word_count_ratio, 3),
        "char_similarity": round(char_similarity, 3),
        "non_empty": non_empty,
    }


# =============================================================================
# Print Settings (--print-settings)
# =============================================================================

def print_settings_snippet(finetuned_url):
    """Print a ready-to-paste settings.json snippet for CLI integration."""
    # Extract host:port from the URL
    parsed = urlparse(finetuned_url)
    host_port = parsed.netloc or "<RUNPOD_POD_IP>:8000"

    # If it's the default localhost, show placeholder instead
    if "localhost" in host_port or "127.0.0.1" in host_port:
        host_port = "<RUNPOD_POD_IP>:8000"

    api_base = f"http://{host_port}/v1"

    snippet = {
        "model_name": FINETUNED_MODEL,
        "api_base_url": api_base,
        "api_key_env_var": "VLLM_API_KEY",
    }

    print()
    print("=" * 60)
    print(" settings.json snippet for chucklesPRIME CLI")
    print("=" * 60)
    print()
    print("Add or update these fields in your settings.json:")
    print()
    print(json.dumps(snippet, indent=2))
    print()
    print("Then set the API key environment variable (vLLM doesn't")
    print("require a real key, but the CLI expects the env var):")
    print()
    print('  export VLLM_API_KEY="not-needed"')
    print()
    print("After that, run the CLI as normal:")
    print()
    print("  chuckles generate input.csv")
    print()
    print("No code changes needed -- the CLI connects to vLLM")
    print("via the same OpenAI-compatible API it already uses.")
    print("=" * 60)


# =============================================================================
# Display Functions
# =============================================================================

def print_separator(char="=", width=70):
    """Print a separator line."""
    print(char * width)


def print_result_row(title, parody_text, scores, label=""):
    """Print a single result row with title, parody, and scores."""
    prefix = f"  [{label}] " if label else "  "
    print(f"{prefix}Title:      {title}")
    # Truncate long parody text for display
    display_text = parody_text[:200] + "..." if len(parody_text) > 200 else parody_text
    print(f"{prefix}Parody:     {display_text}")
    print(f"{prefix}Scores:     word_ratio={scores['word_count_ratio']:.3f}  "
          f"char_sim={scores['char_similarity']:.3f}  "
          f"non_empty={scores['non_empty']:.0f}")


def print_summary_table(finetuned_results, base_results=None):
    """Print a summary comparison table with average scores."""
    print()
    print_separator()
    print("SUMMARY")
    print_separator()
    print()

    # Compute averages for fine-tuned model
    ft_avg = {
        "word_count_ratio": 0.0,
        "char_similarity": 0.0,
        "non_empty": 0.0,
    }
    for _, scores in finetuned_results:
        for key in ft_avg:
            ft_avg[key] += scores[key]
    n = len(finetuned_results)
    for key in ft_avg:
        ft_avg[key] = round(ft_avg[key] / n, 3) if n > 0 else 0.0

    # Header
    if base_results:
        # Compute averages for base model
        base_avg = {
            "word_count_ratio": 0.0,
            "char_similarity": 0.0,
            "non_empty": 0.0,
        }
        for _, scores in base_results:
            for key in base_avg:
                base_avg[key] += scores[key]
        bn = len(base_results)
        for key in base_avg:
            base_avg[key] = round(base_avg[key] / bn, 3) if bn > 0 else 0.0

        print(f"  {'Metric':<20} {'Fine-tuned':>12} {'Base':>12} {'Delta':>12}")
        print(f"  {'-' * 20} {'-' * 12} {'-' * 12} {'-' * 12}")
        for key in ["word_count_ratio", "char_similarity", "non_empty"]:
            delta = ft_avg[key] - base_avg[key]
            sign = "+" if delta >= 0 else ""
            print(f"  {key:<20} {ft_avg[key]:>12.3f} {base_avg[key]:>12.3f} {sign}{delta:>11.3f}")
    else:
        print(f"  {'Metric':<20} {'Fine-tuned':>12}")
        print(f"  {'-' * 20} {'-' * 12}")
        for key in ["word_count_ratio", "char_similarity", "non_empty"]:
            print(f"  {key:<20} {ft_avg[key]:>12.3f}")

    print()

    # Verdict
    print_separator()
    if base_results:
        if ft_avg["char_similarity"] >= base_avg["char_similarity"]:
            print("VERDICT: PASS")
            print("Fine-tuned model has equal or higher character similarity than base.")
        else:
            print("VERDICT: MARGINAL")
            print("Fine-tuned model has lower character similarity than base.")
            print("This may be acceptable if parody quality is subjectively better.")
    else:
        if ft_avg["non_empty"] >= 0.8:
            print("VERDICT: PASS")
            print(f"Fine-tuned model produced valid outputs for "
                  f"{int(ft_avg['non_empty'] * n)}/{n} prompts.")
        else:
            print("VERDICT: FAIL")
            print(f"Fine-tuned model produced valid outputs for only "
                  f"{int(ft_avg['non_empty'] * n)}/{n} prompts.")
    print_separator()

    return ft_avg, base_results and base_avg or None


# =============================================================================
# Main
# =============================================================================

def main():
    """Run the inference quality validation pipeline."""
    parser = argparse.ArgumentParser(
        description="Validate inference quality of fine-tuned vs base model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test fine-tuned model only:
  python validate_inference.py --finetuned-url http://localhost:8000/v1

  # Compare fine-tuned vs base:
  python validate_inference.py \\
      --finetuned-url http://localhost:8000/v1 \\
      --base-url http://localhost:8001/v1

  # Print settings.json snippet:
  python validate_inference.py --print-settings
        """,
    )

    parser.add_argument(
        "--finetuned-url",
        default=FINETUNED_URL,
        help=f"vLLM endpoint URL for fine-tuned model (default: {FINETUNED_URL})",
    )
    parser.add_argument(
        "--base-url",
        default=BASE_URL,
        help="vLLM endpoint URL for base model (optional, enables comparison)",
    )
    parser.add_argument(
        "--finetuned-model",
        default=FINETUNED_MODEL,
        help=f"Served model name for fine-tuned model (default: {FINETUNED_MODEL})",
    )
    parser.add_argument(
        "--base-model",
        default=BASE_MODEL,
        help=f"Served model name for base model (default: {BASE_MODEL})",
    )
    parser.add_argument(
        "--print-settings",
        action="store_true",
        help="Print a settings.json snippet for CLI integration and exit",
    )

    args = parser.parse_args()

    # --print-settings mode: print snippet and exit (no server connection needed)
    if args.print_settings:
        print_settings_snippet(args.finetuned_url)
        sys.exit(0)

    # Banner
    print()
    print_separator()
    print(" chucklesPRIME - Inference Quality Validation")
    print_separator()
    print()
    print(f"  Fine-tuned URL:   {args.finetuned_url}")
    print(f"  Fine-tuned model: {args.finetuned_model}")
    if args.base_url:
        print(f"  Base URL:         {args.base_url}")
        print(f"  Base model:       {args.base_model}")
    else:
        print(f"  Base model:       (not provided -- single-model mode)")
    print(f"  Test titles:      {len(TEST_TITLES)}")
    print()

    # Create OpenAI client(s)
    ft_client = create_client(args.finetuned_url)
    base_client = create_client(args.base_url) if args.base_url else None

    # Verify fine-tuned model is reachable
    print("Checking fine-tuned model connectivity...")
    try:
        models = ft_client.models.list()
        model_ids = [m.id for m in models.data]
        print(f"  Available models: {model_ids}")
        if args.finetuned_model not in model_ids:
            print(f"  WARNING: '{args.finetuned_model}' not in available models.")
            print(f"  Available: {model_ids}")
            print(f"  Requests may fail with 404. Check --served-model-name in vLLM.")
    except Exception as e:
        print(f"  ERROR: Cannot connect to {args.finetuned_url}: {e}")
        print("  Is the vLLM server running? Start with: bash setup_inference.sh --bnb")
        sys.exit(1)

    if base_client:
        print("Checking base model connectivity...")
        try:
            models = base_client.models.list()
            model_ids = [m.id for m in models.data]
            print(f"  Available models: {model_ids}")
        except Exception as e:
            print(f"  WARNING: Cannot connect to {args.base_url}: {e}")
            print("  Continuing without base model comparison.")
            base_client = None

    print()

    # Run inference on all test titles
    finetuned_results = []
    base_results = [] if base_client else None

    for i, title in enumerate(TEST_TITLES):
        print_separator("-")
        print(f"  [{i + 1}/{len(TEST_TITLES)}] Title: '{title}'")
        print_separator("-")

        # Fine-tuned model
        ft_output = query_model(ft_client, args.finetuned_model, title)
        ft_scores = score_parody(title, ft_output)
        finetuned_results.append((ft_output, ft_scores))
        print_result_row(title, ft_output, ft_scores, label="FT")

        # Base model (if available)
        if base_client:
            base_output = query_model(base_client, args.base_model, title)
            base_scores = score_parody(title, base_output)
            base_results.append((base_output, base_scores))
            print_result_row(title, base_output, base_scores, label="Base")

        print()

    # Print summary table
    ft_avg, base_avg = print_summary_table(finetuned_results, base_results)

    # Per-prompt detail table
    print()
    print("Per-prompt scores (fine-tuned):")
    print()
    print(f"  {'#':<4} {'Title':<35} {'WR':>6} {'CS':>6} {'NE':>4}")
    print(f"  {'-' * 4} {'-' * 35} {'-' * 6} {'-' * 6} {'-' * 4}")
    for i, (title, (output, scores)) in enumerate(zip(TEST_TITLES, finetuned_results)):
        print(f"  {i + 1:<4} {title[:35]:<35} "
              f"{scores['word_count_ratio']:>6.3f} "
              f"{scores['char_similarity']:>6.3f} "
              f"{scores['non_empty']:>4.0f}")

    print()
    print("WR = word_count_ratio, CS = char_similarity, NE = non_empty")
    print()

    # Exit code based on non_empty rate
    non_empty_count = sum(1 for _, s in finetuned_results if s["non_empty"] > 0)
    if non_empty_count >= len(TEST_TITLES) * 0.8:
        print("Exit: 0 (PASS)")
        sys.exit(0)
    else:
        print("Exit: 1 (FAIL)")
        sys.exit(1)


if __name__ == "__main__":
    main()
