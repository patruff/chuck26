"""Evaluate trained parody models on phonetic quality.

Generates parodies for a test set of titles and scores them using
phonetic similarity metrics. Outputs a detailed report with pass/fail
status for each title.

Test Criteria:
- Phonetic similarity >= 0.6 for each word substitution
- Overall average score >= 0.65
- Structure preservation (word count matches or close)

Usage:
    # Evaluate a trained model
    python evaluate_parodies.py --model ./parody-model

    # Compare base vs fine-tuned
    python evaluate_parodies.py \
        --model ./parody-model \
        --baseline Qwen/Qwen2.5-1.5B-Instruct

    # Custom test titles
    python evaluate_parodies.py \
        --model ./parody-model \
        --titles "The Matrix,Die Hard,Fight Club"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Phonetic analysis (simplified from the main codebase)
# ---------------------------------------------------------------------------

try:
    import pronouncing
    HAS_PRONOUNCING = True
except ImportError:
    HAS_PRONOUNCING = False
    print("Warning: 'pronouncing' not installed. Phonetic scoring disabled.")


def get_phones(word: str) -> list[str]:
    """Get phonemes for a word using CMU dictionary."""
    if not HAS_PRONOUNCING:
        return []
    phones = pronouncing.phones_for_word(word.lower())
    return phones[0].split() if phones else []


def phonetic_similarity(word1: str, word2: str) -> float:
    """Calculate phonetic similarity between two words (0.0 to 1.0)."""
    if not HAS_PRONOUNCING:
        return 0.5  # Neutral when we can't check

    phones1 = get_phones(word1)
    phones2 = get_phones(word2)

    if not phones1 or not phones2:
        # Fallback to character-level similarity
        return _char_similarity(word1.lower(), word2.lower())

    # Use longest common subsequence ratio
    lcs_len = _lcs_length(phones1, phones2)
    max_len = max(len(phones1), len(phones2))
    return lcs_len / max_len if max_len > 0 else 0.0


def _lcs_length(seq1: list, seq2: list) -> int:
    """Longest common subsequence length."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def _char_similarity(s1: str, s2: str) -> float:
    """Character-level similarity using Levenshtein ratio."""
    if not s1 or not s2:
        return 0.0
    max_len = max(len(s1), len(s2))
    distance = _levenshtein(s1, s2)
    return 1.0 - (distance / max_len)


def _levenshtein(s1: str, s2: str) -> int:
    """Levenshtein edit distance."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

DEFAULT_TEST_TITLES = [
    "The Matrix",
    "Die Hard",
    "Fight Club",
    "Top Gun",
    "Star Wars",
    "Pulp Fiction",
    "The Godfather",
    "Forrest Gump",
    "The Shining",
    "Jurassic Park",
    "Blade Runner",
    "The Terminator",
    "Back to the Future",
    "Ghostbusters",
    "Indiana Jones",
]

SYSTEM_PROMPT = """You are a comedy writer who creates funny parody titles.
Replace words with phonetically similar but humorous alternatives.
The parody should sound similar to the original when spoken aloud."""


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class WordScore:
    original: str
    replacement: str
    score: float
    passed: bool  # score >= 0.6


@dataclass
class ParodyResult:
    input_title: str
    generated_parody: str
    word_scores: list[WordScore] = field(default_factory=list)
    avg_score: float = 0.0
    structure_score: float = 0.0
    passed: bool = False
    error: str | None = None


@dataclass
class EvalReport:
    model_name: str
    results: list[ParodyResult] = field(default_factory=list)
    total: int = 0
    passed: int = 0
    failed: int = 0
    avg_phonetic_score: float = 0.0
    pass_rate: float = 0.0


# ---------------------------------------------------------------------------
# Parody scoring
# ---------------------------------------------------------------------------

def score_parody(original: str, parody: str) -> ParodyResult:
    """Score a parody against its original title."""
    result = ParodyResult(input_title=original, generated_parody=parody)

    # Tokenize into words
    orig_words = re.findall(r'\b\w+\b', original.lower())
    parody_words = re.findall(r'\b\w+\b', parody.lower())

    # Structure score (word count similarity)
    if orig_words:
        result.structure_score = min(len(parody_words), len(orig_words)) / max(len(parody_words), len(orig_words))
    else:
        result.structure_score = 0.0

    # Score each word pair
    scores = []
    for i, orig_word in enumerate(orig_words):
        if i < len(parody_words):
            parody_word = parody_words[i]
            score = phonetic_similarity(orig_word, parody_word)
            passed = score >= 0.6 or orig_word == parody_word  # Same word is fine
            result.word_scores.append(WordScore(
                original=orig_word,
                replacement=parody_word,
                score=score,
                passed=passed,
            ))
            scores.append(score)

    # Average score
    if scores:
        result.avg_score = sum(scores) / len(scores)
    else:
        result.avg_score = 0.0

    # Overall pass: avg >= 0.65 and structure preserved
    result.passed = (
        result.avg_score >= 0.65 and
        result.structure_score >= 0.8 and
        len(result.word_scores) > 0
    )

    return result


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

def load_model(model_path: str, use_4bit: bool = False):
    """Load a model for inference."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check for quantization
    quantization_config = None
    if use_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if not quantization_config else None,
    )

    return model, tokenizer


def generate_parody(model, tokenizer, title: str, max_new_tokens: int = 64) -> str:
    """Generate a parody for a given title."""
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Create a phonetically-sound parody of: '{title}'\n\nRespond with just the parody title, nothing else."},
    ]

    # Apply chat template
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # Fallback for tokenizers without chat template
        prompt = f"{SYSTEM_PROMPT}\n\nUser: Create a parody of '{title}'\n\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the new tokens
    generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    # Clean up: extract just the parody title
    generated = generated.strip()
    # Remove common prefixes
    for prefix in ["Here's", "Here is", "The parody is:", "Parody:", "Answer:"]:
        if generated.lower().startswith(prefix.lower()):
            generated = generated[len(prefix):].strip()
    # Take first line if multi-line
    generated = generated.split("\n")[0].strip()
    # Remove quotes
    generated = generated.strip('"\'')

    return generated


# ---------------------------------------------------------------------------
# Evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate_model(
    model_path: str,
    titles: list[str],
    use_4bit: bool = False,
) -> EvalReport:
    """Evaluate a model on a list of titles."""
    model, tokenizer = load_model(model_path, use_4bit)
    report = EvalReport(model_name=model_path)

    print(f"\nEvaluating on {len(titles)} titles...")
    print("-" * 60)

    for title in titles:
        try:
            parody = generate_parody(model, tokenizer, title)
            result = score_parody(title, parody)
        except Exception as e:
            result = ParodyResult(
                input_title=title,
                generated_parody="",
                error=str(e),
            )

        report.results.append(result)

        # Print result
        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(f"{status}  {title:25} → {result.generated_parody:25} (avg: {result.avg_score:.2f})")

    # Compute summary stats
    report.total = len(report.results)
    report.passed = sum(1 for r in report.results if r.passed)
    report.failed = report.total - report.passed
    scores = [r.avg_score for r in report.results if not r.error]
    report.avg_phonetic_score = sum(scores) / len(scores) if scores else 0.0
    report.pass_rate = report.passed / report.total if report.total > 0 else 0.0

    return report


def print_report(report: EvalReport):
    """Print a detailed evaluation report."""
    print("\n" + "=" * 60)
    print(f"EVALUATION REPORT: {report.model_name}")
    print("=" * 60)
    print(f"  Total titles:     {report.total}")
    print(f"  Passed:           {report.passed}")
    print(f"  Failed:           {report.failed}")
    print(f"  Pass rate:        {report.pass_rate:.1%}")
    print(f"  Avg phonetic:     {report.avg_phonetic_score:.3f}")
    print()

    # Show failures in detail
    failures = [r for r in report.results if not r.passed]
    if failures:
        print("Failed cases:")
        for r in failures:
            print(f"  {r.input_title} → {r.generated_parody}")
            if r.word_scores:
                for ws in r.word_scores:
                    status = "✓" if ws.passed else "✗"
                    print(f"    {status} {ws.original} → {ws.replacement} ({ws.score:.2f})")
            if r.error:
                print(f"    ERROR: {r.error}")
        print()


def save_report(report: EvalReport, output_path: str):
    """Save report as JSON."""
    data = {
        "model_name": report.model_name,
        "total": report.total,
        "passed": report.passed,
        "failed": report.failed,
        "pass_rate": report.pass_rate,
        "avg_phonetic_score": report.avg_phonetic_score,
        "results": [
            {
                "input_title": r.input_title,
                "generated_parody": r.generated_parody,
                "avg_score": r.avg_score,
                "structure_score": r.structure_score,
                "passed": r.passed,
                "error": r.error,
                "word_scores": [
                    {
                        "original": ws.original,
                        "replacement": ws.replacement,
                        "score": ws.score,
                        "passed": ws.passed,
                    }
                    for ws in r.word_scores
                ],
            }
            for r in report.results
        ],
    }
    Path(output_path).write_text(json.dumps(data, indent=2))
    print(f"Report saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate parody model quality",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained model or HuggingFace model ID",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Optional baseline model to compare against",
    )
    parser.add_argument(
        "--titles",
        default=None,
        help="Comma-separated list of titles to test (default: built-in list)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save JSON report",
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Use 4-bit quantization for inference",
    )
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=0.7,
        help="Minimum pass rate to consider evaluation successful",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Parse titles
    if args.titles:
        titles = [t.strip() for t in args.titles.split(",")]
    else:
        titles = DEFAULT_TEST_TITLES

    # Evaluate main model
    report = evaluate_model(args.model, titles, use_4bit=args.use_4bit)
    print_report(report)

    # Evaluate baseline if provided
    baseline_report = None
    if args.baseline:
        print("\n" + "=" * 60)
        print("BASELINE COMPARISON")
        print("=" * 60)
        baseline_report = evaluate_model(args.baseline, titles, use_4bit=args.use_4bit)
        print_report(baseline_report)

        # Compare
        print("\nComparison:")
        print(f"  Model pass rate:    {report.pass_rate:.1%}")
        print(f"  Baseline pass rate: {baseline_report.pass_rate:.1%}")
        diff = report.pass_rate - baseline_report.pass_rate
        if diff > 0:
            print(f"  Improvement:        +{diff:.1%} ✓")
        elif diff < 0:
            print(f"  Regression:         {diff:.1%} ✗")
        else:
            print(f"  No change")

    # Save report
    if args.output:
        save_report(report, args.output)

    # Exit with appropriate code
    if report.pass_rate >= args.min_pass_rate:
        print(f"\n✓ Evaluation PASSED (pass rate {report.pass_rate:.1%} >= {args.min_pass_rate:.1%})")
        sys.exit(0)
    else:
        print(f"\n✗ Evaluation FAILED (pass rate {report.pass_rate:.1%} < {args.min_pass_rate:.1%})")
        sys.exit(1)


if __name__ == "__main__":
    main()
