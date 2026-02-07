"""Training report generator with timing, cost, and model comparison.

Generates a comprehensive report after DPO training that includes:
- Training time and estimated cost
- Model details and HuggingFace link
- Side-by-side comparison of base vs fine-tuned model outputs
- Evaluation metrics and pass rates
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


# GPU cost per hour (approximate RunPod prices)
GPU_COSTS = {
    "NVIDIA GeForce RTX 3090": 0.22,
    "NVIDIA GeForce RTX 4090": 0.44,
    "NVIDIA A40": 0.39,
    "NVIDIA L40": 0.49,
    "NVIDIA A100 40GB PCIe": 1.09,
    "NVIDIA A100 80GB PCIe": 1.69,
    # Short names
    "rtx3090": 0.22,
    "rtx4090": 0.44,
    "a40": 0.39,
    "l40": 0.49,
    "a100-40": 1.09,
    "a100-80": 1.69,
}


@dataclass
class ComparisonExample:
    """Single comparison between base and fine-tuned model."""
    input_title: str
    base_output: str
    finetuned_output: str
    base_score: float = 0.0
    finetuned_score: float = 0.0
    improvement: float = 0.0


@dataclass
class TrainingReport:
    """Comprehensive training report."""
    # Model info
    base_model: str = ""
    output_model: str = ""
    huggingface_url: str = ""

    # Dataset info
    dataset: str = ""
    dataset_size: int = 0

    # Training config
    epochs: int = 0
    batch_size: int = 0
    learning_rate: float = 0.0
    lora_r: int = 0
    lora_alpha: int = 0
    use_4bit: bool = False
    use_8bit: bool = False

    # GPU info
    gpu_type: str = ""
    gpu_cost_per_hour: float = 0.0

    # Timing
    start_time: str = ""
    end_time: str = ""
    training_duration_seconds: float = 0.0
    training_duration_human: str = ""

    # Cost
    estimated_cost: float = 0.0

    # Evaluation results
    eval_total: int = 0
    eval_passed: int = 0
    eval_pass_rate: float = 0.0
    eval_avg_score: float = 0.0

    # Comparison examples
    comparison_examples: list[ComparisonExample] = field(default_factory=list)

    # Debug log
    debug_log: list[str] = field(default_factory=list)

    def add_log(self, message: str):
        """Add a timestamped debug log entry."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] {message}"
        self.debug_log.append(entry)
        print(entry)

    def set_timing(self, start: float, end: float):
        """Set timing information from timestamps."""
        self.start_time = datetime.fromtimestamp(start).strftime("%Y-%m-%d %H:%M:%S")
        self.end_time = datetime.fromtimestamp(end).strftime("%Y-%m-%d %H:%M:%S")
        self.training_duration_seconds = end - start

        # Human-readable duration
        hours = int(self.training_duration_seconds // 3600)
        minutes = int((self.training_duration_seconds % 3600) // 60)
        seconds = int(self.training_duration_seconds % 60)

        if hours > 0:
            self.training_duration_human = f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            self.training_duration_human = f"{minutes}m {seconds}s"
        else:
            self.training_duration_human = f"{seconds}s"

    def calculate_cost(self):
        """Calculate estimated cost based on GPU and duration."""
        hours = self.training_duration_seconds / 3600
        cost_per_hour = GPU_COSTS.get(self.gpu_type, 0.0)
        if cost_per_hour == 0:
            # Try to match partial GPU name
            for gpu_name, cost in GPU_COSTS.items():
                if gpu_name.lower() in self.gpu_type.lower():
                    cost_per_hour = cost
                    break

        self.gpu_cost_per_hour = cost_per_hour
        self.estimated_cost = round(hours * cost_per_hour, 2)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert ComparisonExample objects
        data["comparison_examples"] = [asdict(ex) for ex in self.comparison_examples]
        return data

    def save(self, path: str | Path):
        """Save report to JSON file."""
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Report saved to: {path}")

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = []

        # Header
        lines.append("# Training Report")
        lines.append("")
        lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Model Info
        lines.append("## Model Information")
        lines.append("")
        lines.append(f"| Property | Value |")
        lines.append(f"|----------|-------|")
        lines.append(f"| Base Model | `{self.base_model}` |")
        lines.append(f"| Output Model | `{self.output_model}` |")
        if self.huggingface_url:
            lines.append(f"| HuggingFace | [{self.output_model}]({self.huggingface_url}) |")
        lines.append(f"| Dataset | `{self.dataset}` |")
        lines.append(f"| Dataset Size | {self.dataset_size} examples |")
        lines.append("")

        # Training Config
        lines.append("## Training Configuration")
        lines.append("")
        lines.append(f"| Parameter | Value |")
        lines.append(f"|-----------|-------|")
        lines.append(f"| Epochs | {self.epochs} |")
        lines.append(f"| Batch Size | {self.batch_size} |")
        lines.append(f"| Learning Rate | {self.learning_rate} |")
        lines.append(f"| LoRA Rank | {self.lora_r} |")
        lines.append(f"| LoRA Alpha | {self.lora_alpha} |")
        lines.append(f"| 4-bit Quantization | {'Yes' if self.use_4bit else 'No'} |")
        lines.append(f"| 8-bit Quantization | {'Yes' if self.use_8bit else 'No'} |")
        lines.append("")

        # Timing & Cost
        lines.append("## Timing & Cost")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| GPU | `{self.gpu_type}` |")
        lines.append(f"| GPU Cost | ${self.gpu_cost_per_hour:.2f}/hr |")
        lines.append(f"| Start Time | {self.start_time} |")
        lines.append(f"| End Time | {self.end_time} |")
        lines.append(f"| Duration | **{self.training_duration_human}** |")
        lines.append(f"| Estimated Cost | **${self.estimated_cost:.2f}** |")
        lines.append("")

        # Evaluation Results
        lines.append("## Evaluation Results")
        lines.append("")
        if self.eval_total > 0:
            status = "PASSED" if self.eval_pass_rate >= 0.7 else "FAILED"
            emoji = "✅" if self.eval_pass_rate >= 0.7 else "❌"
            lines.append(f"| Metric | Value |")
            lines.append(f"|--------|-------|")
            lines.append(f"| Total Titles | {self.eval_total} |")
            lines.append(f"| Passed | {self.eval_passed} |")
            lines.append(f"| Pass Rate | **{self.eval_pass_rate:.1%}** |")
            lines.append(f"| Avg Phonetic Score | {self.eval_avg_score:.3f} |")
            lines.append(f"| Status | {emoji} **{status}** |")
        else:
            lines.append("*Evaluation not run*")
        lines.append("")

        # Comparison Examples
        if self.comparison_examples:
            lines.append("## Model Comparison (Base vs Fine-tuned)")
            lines.append("")
            lines.append("| Input Title | Base Model | Fine-tuned | Base Score | FT Score | Δ |")
            lines.append("|-------------|------------|------------|------------|----------|---|")

            for ex in self.comparison_examples:
                delta = ex.improvement
                delta_str = f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}"
                emoji = "📈" if delta > 0 else ("📉" if delta < 0 else "➖")
                lines.append(
                    f"| {ex.input_title} | {ex.base_output} | {ex.finetuned_output} | "
                    f"{ex.base_score:.2f} | {ex.finetuned_score:.2f} | {emoji} {delta_str} |"
                )
            lines.append("")

            # Summary
            if self.comparison_examples:
                avg_improvement = sum(ex.improvement for ex in self.comparison_examples) / len(self.comparison_examples)
                improved = sum(1 for ex in self.comparison_examples if ex.improvement > 0)
                lines.append(f"**Summary**: {improved}/{len(self.comparison_examples)} examples improved, "
                           f"average improvement: {avg_improvement:+.2f}")
            lines.append("")

        # Debug Log (last 20 entries)
        if self.debug_log:
            lines.append("## Debug Log (Last 20 Entries)")
            lines.append("")
            lines.append("```")
            for entry in self.debug_log[-20:]:
                lines.append(entry)
            lines.append("```")
            lines.append("")

        return "\n".join(lines)

    def save_markdown(self, path: str | Path):
        """Save markdown report."""
        path = Path(path)
        with open(path, "w") as f:
            f.write(self.to_markdown())
        print(f"Markdown report saved to: {path}")

    def print_summary(self):
        """Print a summary to stdout."""
        print("\n" + "=" * 70)
        print("TRAINING REPORT SUMMARY")
        print("=" * 70)
        print(f"  Model:      {self.output_model}")
        print(f"  Base:       {self.base_model}")
        print(f"  Dataset:    {self.dataset} ({self.dataset_size} examples)")
        print(f"  GPU:        {self.gpu_type}")
        print(f"  Duration:   {self.training_duration_human}")
        print(f"  Cost:       ${self.estimated_cost:.2f}")
        print("-" * 70)
        if self.eval_total > 0:
            status = "PASSED" if self.eval_pass_rate >= 0.7 else "FAILED"
            print(f"  Eval:       {self.eval_passed}/{self.eval_total} passed ({self.eval_pass_rate:.1%}) - {status}")
            print(f"  Avg Score:  {self.eval_avg_score:.3f}")
        print("-" * 70)
        if self.comparison_examples:
            print("  Comparison Examples:")
            for ex in self.comparison_examples:
                delta = f"+{ex.improvement:.2f}" if ex.improvement > 0 else f"{ex.improvement:.2f}"
                print(f"    {ex.input_title}")
                print(f"      Base:      {ex.base_output} (score: {ex.base_score:.2f})")
                print(f"      Finetuned: {ex.finetuned_output} (score: {ex.finetuned_score:.2f}, {delta})")
        print("=" * 70)
        if self.huggingface_url:
            print(f"\nModel: {self.huggingface_url}")


def generate_comparison_examples(
    base_model_path: str,
    finetuned_model_path: str,
    test_titles: list[str] | None = None,
    num_examples: int = 5,
) -> list[ComparisonExample]:
    """Generate comparison examples between base and fine-tuned models.

    Args:
        base_model_path: Path or HF ID of base model
        finetuned_model_path: Path to fine-tuned model
        test_titles: List of titles to test (uses defaults if None)
        num_examples: Number of examples to generate

    Returns:
        List of ComparisonExample objects
    """
    # Import here to avoid loading torch at module level
    from evaluate_parodies import (
        DEFAULT_TEST_TITLES,
        generate_parody,
        load_model,
        score_parody,
    )

    if test_titles is None:
        test_titles = DEFAULT_TEST_TITLES[:num_examples]
    else:
        test_titles = test_titles[:num_examples]

    examples = []

    # Load base model
    print(f"\nLoading base model: {base_model_path}")
    try:
        base_model, base_tokenizer = load_model(base_model_path, use_4bit=True)
    except Exception as e:
        print(f"Warning: Could not load base model: {e}")
        base_model, base_tokenizer = None, None

    # Load fine-tuned model
    print(f"Loading fine-tuned model: {finetuned_model_path}")
    ft_model, ft_tokenizer = load_model(finetuned_model_path, use_4bit=True)

    print(f"\nGenerating {len(test_titles)} comparison examples...")

    for title in test_titles:
        print(f"  Processing: {title}")

        # Generate with base model
        if base_model is not None:
            try:
                base_output = generate_parody(base_model, base_tokenizer, title)
                base_result = score_parody(title, base_output)
                base_score = base_result.avg_score
            except Exception as e:
                print(f"    Base model error: {e}")
                base_output = "[error]"
                base_score = 0.0
        else:
            base_output = "[not loaded]"
            base_score = 0.0

        # Generate with fine-tuned model
        try:
            ft_output = generate_parody(ft_model, ft_tokenizer, title)
            ft_result = score_parody(title, ft_output)
            ft_score = ft_result.avg_score
        except Exception as e:
            print(f"    Fine-tuned model error: {e}")
            ft_output = "[error]"
            ft_score = 0.0

        example = ComparisonExample(
            input_title=title,
            base_output=base_output,
            finetuned_output=ft_output,
            base_score=base_score,
            finetuned_score=ft_score,
            improvement=round(ft_score - base_score, 3),
        )
        examples.append(example)

        print(f"    Base: {base_output} ({base_score:.2f})")
        print(f"    FT:   {ft_output} ({ft_score:.2f})")

    # Clean up GPU memory
    try:
        import torch
        del ft_model, ft_tokenizer
        if base_model is not None:
            del base_model, base_tokenizer
        torch.cuda.empty_cache()
    except Exception:
        pass

    return examples


# ---------------------------------------------------------------------------
# Standalone usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate training report")
    parser.add_argument("--base-model", required=True, help="Base model path/ID")
    parser.add_argument("--finetuned-model", required=True, help="Fine-tuned model path")
    parser.add_argument("--output", default="training-report.json", help="Output JSON path")
    parser.add_argument("--markdown", default="training-report.md", help="Output markdown path")
    parser.add_argument("--examples", type=int, default=5, help="Number of comparison examples")

    args = parser.parse_args()

    # Create report
    report = TrainingReport(
        base_model=args.base_model,
        output_model=args.finetuned_model,
    )

    # Generate comparison examples
    print("Generating comparison examples...")
    report.comparison_examples = generate_comparison_examples(
        args.base_model,
        args.finetuned_model,
        num_examples=args.examples,
    )

    # Save reports
    report.save(args.output)
    report.save_markdown(args.markdown)
    report.print_summary()
