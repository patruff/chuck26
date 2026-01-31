#!/usr/bin/env python3
"""
Popular Movies Test - Tests parody generation with 10 popular movies
Also captures phonetics tool calls and reasoning for RLVR dataset creation.

Usage:
    python test_popular_movies.py [--limit N] [--model MODEL] [--output-dir DIR]

Example:
    python test_popular_movies.py --limit 3  # Test with first 3 movies
    python test_popular_movies.py            # Test all 10 movies
"""

import os
import sys
import csv
import json
import re
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# =============================================================================
# RLVR Chat Template Configuration
# =============================================================================
# Customize these tags for your training setup. Different base models may
# benefit from different tag formats. DeepSeek uses <think>/<think>, but
# you can customize for your specific RLVR training pipeline.

@dataclass
class RLVRTemplateTags:
    """Configurable tags for RLVR training format."""
    # Reasoning/thinking tags
    reasoning_start: str = "<start_working_out>"
    reasoning_end: str = "<end_working_out>"

    # Solution/answer tags
    solution_start: str = "<SOLUTION>"
    solution_end: str = "</SOLUTION>"

    # Legacy tags (for backward compatibility with existing data)
    legacy_think_start: str = "<think>"
    legacy_think_end: str = "</think>"

    def get_system_prompt(self) -> str:
        """Generate a system prompt that instructs the model to use these tags."""
        return f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {self.reasoning_start} and {self.reasoning_end}.
Then, provide your solution between {self.solution_start} and {self.solution_end}."""

    def format_reasoning(self, reasoning_text: str) -> str:
        """Format reasoning text with the configured tags."""
        return f"{self.reasoning_start}\n{reasoning_text}\n{self.reasoning_end}"

    def format_solution(self, solution_text: str) -> str:
        """Format solution text with the configured tags."""
        return f"{self.solution_start}{solution_text}{self.solution_end}"

    def format_full_response(self, reasoning: str, solution: str) -> str:
        """Format a complete response with reasoning and solution."""
        return f"""{self.reasoning_start}
{reasoning}
{self.reasoning_end}

{self.solution_start}{solution}{self.solution_end}"""


# Default template tags - can be overridden via command line or config
DEFAULT_TEMPLATE = RLVRTemplateTags()

# Alternative templates for different training setups
DEEPSEEK_TEMPLATE = RLVRTemplateTags(
    reasoning_start="<think>",
    reasoning_end="</think>",
    solution_start="",
    solution_end=""
)

TEMPLATES = {
    "default": DEFAULT_TEMPLATE,
    "deepseek": DEEPSEEK_TEMPLATE,
    "custom": None  # Will be created from CLI args if needed
}

# 10 Popular movies for testing - these are well-known titles with good parody potential
POPULAR_MOVIES = [
    "The Matrix",
    "Die Hard",
    "Fight Club",
    "Star Wars",
    "Top Gun",
    "The Godfather",
    "Pulp Fiction",
    "Forrest Gump",
    "The Shining",
    "Jurassic Park",
]


@dataclass
class ToolCall:
    """Represents a single tool call made during generation."""
    tool_name: str
    arguments: Dict[str, Any]
    result: Any
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ParodyAttempt:
    """Represents a single parody attempt with its scores."""
    parody_text: str
    phonetic_checks: Dict[str, float]  # word -> score mapping
    humor_rating: Optional[int] = None
    explanation: str = ""


@dataclass
class RLVRDataPoint:
    """
    A structured data point for RLVR training.
    Contains input, reasoning trace, tool calls, and output quality signals.
    """
    # Input
    input_title: str

    # Reasoning trace (the full thinking process)
    thinking_trace: str = ""

    # Tool calls made during generation (for learning tool use)
    tool_calls: List[ToolCall] = field(default_factory=list)

    # All parody attempts (for learning from multiple tries)
    attempts: List[ParodyAttempt] = field(default_factory=list)

    # Final output
    final_parody: str = ""
    final_reasoning: str = ""

    # Quality signals (for reward modeling)
    all_phonetic_scores_valid: bool = False  # All scores > 0.6
    average_phonetic_score: float = 0.0
    humor_rating: Optional[int] = None

    # Metadata
    model_name: str = ""
    generation_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    raw_output: str = ""

    # Quality classification (can be labeled later for DPO/RLVR)
    quality_label: Optional[str] = None  # "good", "bad", or None (unlabeled)

    # Template information for RLVR training
    template_name: str = "default"

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "input_title": self.input_title,
            "thinking_trace": self.thinking_trace,
            "tool_calls": [
                {
                    "tool_name": tc.tool_name,
                    "arguments": tc.arguments,
                    "result": tc.result,
                    "timestamp": tc.timestamp
                }
                for tc in self.tool_calls
            ],
            "attempts": [
                {
                    "parody_text": a.parody_text,
                    "phonetic_checks": a.phonetic_checks,
                    "humor_rating": a.humor_rating,
                    "explanation": a.explanation
                }
                for a in self.attempts
            ],
            "final_parody": self.final_parody,
            "final_reasoning": self.final_reasoning,
            "all_phonetic_scores_valid": self.all_phonetic_scores_valid,
            "average_phonetic_score": self.average_phonetic_score,
            "humor_rating": self.humor_rating,
            "model_name": self.model_name,
            "generation_timestamp": self.generation_timestamp,
            "raw_output": self.raw_output,
            "quality_label": self.quality_label,
            "template_name": self.template_name
        }

    def to_rlvr_training_format(self, template: RLVRTemplateTags) -> Dict:
        """
        Convert to RLVR training format with proper tags.

        This format is suitable for training base models with RLVR.
        """
        # Format tool calls as part of reasoning
        tool_trace = ""
        if self.tool_calls:
            tool_trace = "\n\nTool calls made:\n"
            for tc in self.tool_calls:
                args = tc.arguments
                tool_trace += f"- {tc.tool_name}(\"{args.get('word1', '')}\", \"{args.get('word2', '')}\") → {tc.result}\n"

        # Combine reasoning
        full_reasoning = self.thinking_trace
        if tool_trace:
            full_reasoning += tool_trace

        return {
            "system_prompt": template.get_system_prompt(),
            "prompt": f"Create a funny parody of the movie title '{self.input_title}'. "
                     f"Use phonetic similarity checking to verify your word choices sound similar.",
            "response": template.format_full_response(full_reasoning, self.final_parody),
            "reasoning_only": template.format_reasoning(full_reasoning),
            "solution_only": template.format_solution(self.final_parody),
            "rewards": {
                "phonetic_validity": self.all_phonetic_scores_valid,
                "average_phonetic_score": self.average_phonetic_score,
                "humor_rating": self.humor_rating,
                "tool_usage_count": len(self.tool_calls),
                "has_reasoning": bool(self.thinking_trace)
            },
            "verifiable_checks": [
                {
                    "type": "phonetic_score",
                    "word1": tc.arguments.get("word1", ""),
                    "word2": tc.arguments.get("word2", ""),
                    "score": tc.result,
                    "passed": tc.result > 0.6 if isinstance(tc.result, (int, float)) else False
                }
                for tc in self.tool_calls
            ]
        }


class EnhancedOutputCapture:
    """
    Enhanced output capture that extracts structured data for RLVR training.
    Captures tool calls, reasoning traces, and quality signals.
    """

    def __init__(self, title: str, model_name: str, template: RLVRTemplateTags = None):
        self.title = title
        self.model_name = model_name
        self.template = template or DEFAULT_TEMPLATE
        self.data_point = RLVRDataPoint(
            input_title=title,
            model_name=model_name,
            template_name="custom" if template else "default"
        )
        self.raw_outputs: List[str] = []

    def extract_tool_calls(self, text: str) -> List[ToolCall]:
        """Extract tool calls from the output text."""
        tool_calls = []

        # Pattern to match word_phone_tool calls and their results
        # Matches patterns like: word_phone_tool("Running", "Cunning") -> 0.85
        tool_pattern = r'word_phone_tool\s*\(\s*["\']([^"\']+)["\']\s*,\s*["\']([^"\']+)["\']\s*\).*?(?:->|:|\=)\s*([\d.]+)'

        matches = re.finditer(tool_pattern, text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            word1, word2, score = match.groups()
            try:
                score_float = float(score)
                tool_calls.append(ToolCall(
                    tool_name="word_phone_tool",
                    arguments={"word1": word1, "word2": word2},
                    result=score_float
                ))
            except ValueError:
                pass

        # Also try to extract from formatted output lines
        # Matches: - Original "word1" vs "replacement1": 0.85
        alt_pattern = r'[Oo]riginal\s*["\']?(\w+)["\']?\s*(?:vs|→|->)\s*["\']?(\w+)["\']?\s*:\s*([\d.]+)'
        alt_matches = re.finditer(alt_pattern, text)
        for match in alt_matches:
            word1, word2, score = match.groups()
            try:
                score_float = float(score)
                # Avoid duplicates
                if not any(tc.arguments.get("word1") == word1 and tc.arguments.get("word2") == word2
                          for tc in tool_calls):
                    tool_calls.append(ToolCall(
                        tool_name="word_phone_tool",
                        arguments={"word1": word1, "word2": word2},
                        result=score_float
                    ))
            except ValueError:
                pass

        return tool_calls

    def extract_attempts(self, text: str) -> List[ParodyAttempt]:
        """Extract individual parody attempts from the output."""
        attempts = []

        # Pattern to match attempts
        attempt_pattern = r'### Attempt (\d+):\s*\n\*\*"?([^"*\n]+)"?\*\*'

        matches = re.finditer(attempt_pattern, text, re.DOTALL)
        for match in matches:
            attempt_num, parody_text = match.groups()
            parody_text = parody_text.strip()

            # Skip template placeholders
            if '[' in parody_text:
                continue

            # Try to extract humor rating for this attempt
            humor_pattern = rf'### Attempt {attempt_num}:.*?[Hh]umor\s*[Rr]ating:\s*(\d+)/10'
            humor_match = re.search(humor_pattern, text, re.DOTALL)
            humor_rating = int(humor_match.group(1)) if humor_match else None

            # Try to extract explanation
            explanation_pattern = rf'### Attempt {attempt_num}:.*?[Ww]hy it\'?s? funny:\s*([^\n]+)'
            exp_match = re.search(explanation_pattern, text, re.DOTALL)
            explanation = exp_match.group(1).strip() if exp_match else ""

            attempts.append(ParodyAttempt(
                parody_text=parody_text,
                phonetic_checks={},  # Will be filled from tool calls
                humor_rating=humor_rating,
                explanation=explanation
            ))

        return attempts

    def extract_thinking_trace(self, text: str) -> str:
        """Extract the thinking trace from reasoning tags (supports multiple formats)."""
        # Try the configured template tags first
        start_tag = re.escape(self.template.reasoning_start)
        end_tag = re.escape(self.template.reasoning_end)
        pattern = rf'{start_tag}(.*?){end_tag}'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Fall back to legacy <think> tags for backward compatibility
        legacy_pattern = r'<think>(.*?)</think>'
        legacy_match = re.search(legacy_pattern, text, re.DOTALL)
        if legacy_match:
            return legacy_match.group(1).strip()

        # Try other common formats
        alt_patterns = [
            r'<start_working_out>(.*?)<end_working_out>',
            r'<reasoning>(.*?)</reasoning>',
            r'<thought>(.*?)</thought>',
        ]
        for alt_pattern in alt_patterns:
            alt_match = re.search(alt_pattern, text, re.DOTALL)
            if alt_match:
                return alt_match.group(1).strip()

        return ""

    def extract_final_parody(self, text: str) -> str:
        """Extract the final chosen parody."""
        pattern = r'### Final Chosen Parody:.*?\n\*\*"?([^"*\n]+)"?\*\*'
        matches = re.findall(pattern, text, re.DOTALL)

        # Filter out template placeholders
        valid_matches = [m.strip() for m in matches if '[' not in m]
        return valid_matches[-1] if valid_matches else ""

    def extract_final_reasoning(self, text: str) -> str:
        """Extract the final reasoning section."""
        pattern = r'### Final Reasoning:(.*?)(?=###|\Z)'
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ""

    def calculate_quality_signals(self):
        """Calculate quality signals from the extracted data."""
        all_scores = []

        for tc in self.data_point.tool_calls:
            if tc.tool_name == "word_phone_tool" and isinstance(tc.result, (int, float)):
                all_scores.append(float(tc.result))

        if all_scores:
            self.data_point.average_phonetic_score = sum(all_scores) / len(all_scores)
            self.data_point.all_phonetic_scores_valid = all(s > 0.6 for s in all_scores)

        # Get humor rating from best attempt or final
        if self.data_point.attempts:
            ratings = [a.humor_rating for a in self.data_point.attempts if a.humor_rating]
            if ratings:
                self.data_point.humor_rating = max(ratings)

    def process_output(self, raw_output: str):
        """Process the raw output and extract all structured data."""
        self.raw_outputs.append(raw_output)
        self.data_point.raw_output = raw_output

        # Extract components
        self.data_point.thinking_trace = self.extract_thinking_trace(raw_output)
        self.data_point.tool_calls = self.extract_tool_calls(raw_output)
        self.data_point.attempts = self.extract_attempts(raw_output)
        self.data_point.final_parody = self.extract_final_parody(raw_output)
        self.data_point.final_reasoning = self.extract_final_reasoning(raw_output)

        # Calculate quality signals
        self.calculate_quality_signals()

        return self.data_point


def generate_parody_with_capture(title: str, model_name: str, api_key: str, output_dir: str) -> RLVRDataPoint:
    """
    Generate a parody and capture all data for RLVR training.
    """
    from generate_parody import generate_parody

    # Create enhanced capture
    capture = EnhancedOutputCapture(title, model_name)

    # Generate parody
    result = generate_parody(
        title=title,
        model_name=model_name,
        api_key=api_key,
        output_dir=output_dir
    )

    # Process and extract structured data
    data_point = capture.process_output(result)

    return data_point


def save_rlvr_dataset(data_points: List[RLVRDataPoint], output_path: str, template: RLVRTemplateTags = None):
    """
    Save RLVR dataset in multiple formats for different use cases.

    Args:
        data_points: List of RLVRDataPoint objects to save
        output_path: Directory to save output files
        template: Template tags to use for formatting (default: DEFAULT_TEMPLATE)
    """
    template = template or DEFAULT_TEMPLATE
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Save as JSON Lines (good for streaming/loading)
    jsonl_path = output_path / f"rlvr_dataset_{timestamp}.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for dp in data_points:
            f.write(json.dumps(dp.to_dict(), ensure_ascii=False) + '\n')
    logging.info(f"Saved JSONL dataset: {jsonl_path}")

    # 2. Save as full JSON (good for inspection)
    json_path = output_path / f"rlvr_dataset_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump([dp.to_dict() for dp in data_points], f, indent=2, ensure_ascii=False)
    logging.info(f"Saved JSON dataset: {json_path}")

    # 3. Save simplified CSV for quick review
    csv_path = output_path / f"rlvr_summary_{timestamp}.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'input_title', 'final_parody', 'avg_phonetic_score',
            'all_scores_valid', 'humor_rating', 'num_tool_calls',
            'num_attempts', 'quality_label'
        ])
        for dp in data_points:
            writer.writerow([
                dp.input_title,
                dp.final_parody,
                f"{dp.average_phonetic_score:.2f}",
                dp.all_phonetic_scores_valid,
                dp.humor_rating or "N/A",
                len(dp.tool_calls),
                len(dp.attempts),
                dp.quality_label or "unlabeled"
            ])
    logging.info(f"Saved CSV summary: {csv_path}")

    # 4. Save tool calls separately (for tool-use training)
    tool_calls_path = output_path / f"tool_calls_{timestamp}.jsonl"
    with open(tool_calls_path, 'w', encoding='utf-8') as f:
        for dp in data_points:
            for tc in dp.tool_calls:
                entry = {
                    "input_title": dp.input_title,
                    "tool_name": tc.tool_name,
                    "arguments": tc.arguments,
                    "result": tc.result,
                    "context": f"Generating parody for '{dp.input_title}'"
                }
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    logging.info(f"Saved tool calls: {tool_calls_path}")

    # 5. Save reasoning traces (for reasoning training)
    reasoning_path = output_path / f"reasoning_traces_{timestamp}.jsonl"
    with open(reasoning_path, 'w', encoding='utf-8') as f:
        for dp in data_points:
            if dp.thinking_trace:
                entry = {
                    "input": f"Create a funny parody of the movie title '{dp.input_title}'",
                    "reasoning": dp.thinking_trace,
                    "output": dp.final_parody,
                    "quality_signals": {
                        "phonetic_valid": dp.all_phonetic_scores_valid,
                        "avg_score": dp.average_phonetic_score,
                        "humor_rating": dp.humor_rating
                    }
                }
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    logging.info(f"Saved reasoning traces: {reasoning_path}")

    # 6. Save RLVR training format (with configurable tags for base model training)
    rlvr_training_path = output_path / f"rlvr_training_{timestamp}.jsonl"
    with open(rlvr_training_path, 'w', encoding='utf-8') as f:
        for dp in data_points:
            if dp.final_parody and not dp.final_parody.startswith("ERROR"):
                entry = dp.to_rlvr_training_format(template)
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    logging.info(f"Saved RLVR training format: {rlvr_training_path}")

    # 7. Save template configuration for reference
    template_config_path = output_path / f"template_config_{timestamp}.json"
    with open(template_config_path, 'w', encoding='utf-8') as f:
        json.dump({
            "reasoning_start": template.reasoning_start,
            "reasoning_end": template.reasoning_end,
            "solution_start": template.solution_start,
            "solution_end": template.solution_end,
            "system_prompt": template.get_system_prompt(),
            "timestamp": timestamp
        }, f, indent=2)
    logging.info(f"Saved template config: {template_config_path}")

    return {
        "jsonl": str(jsonl_path),
        "json": str(json_path),
        "csv": str(csv_path),
        "tool_calls": str(tool_calls_path),
        "reasoning": str(reasoning_path),
        "rlvr_training": str(rlvr_training_path),
        "template_config": str(template_config_path)
    }


def run_popular_movies_test(
    limit: Optional[int] = None,
    model_name: str = "qwen-3-32b",
    api_key: Optional[str] = None,
    output_dir: str = "./rlvr_output",
    template: RLVRTemplateTags = None
):
    """
    Run the popular movies test and generate RLVR training data.

    Args:
        limit: Number of movies to test (default: all 10)
        model_name: Cerebras model to use
        api_key: Cerebras API key
        output_dir: Directory for output files
        template: Template tags to use for RLVR output formatting
    """
    template = template or DEFAULT_TEMPLATE
    movies = POPULAR_MOVIES[:limit] if limit else POPULAR_MOVIES

    logging.info(f"\n{'='*80}")
    logging.info("POPULAR MOVIES TEST - RLVR Data Generation")
    logging.info(f"{'='*80}")
    logging.info(f"Movies to test: {len(movies)}")
    logging.info(f"Model: {model_name}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Template tags: {template.reasoning_start}...{template.reasoning_end}")
    logging.info(f"{'='*80}\n")

    # Process each movie
    data_points: List[RLVRDataPoint] = []

    for i, movie in enumerate(movies, 1):
        logging.info(f"\n{'='*60}")
        logging.info(f"[{i}/{len(movies)}] Processing: {movie}")
        logging.info(f"{'='*60}")

        try:
            movie_output_dir = f"{output_dir}/individual/{i:02d}_{movie.replace(' ', '_')}"

            data_point = generate_parody_with_capture(
                title=movie,
                model_name=model_name,
                api_key=api_key,
                output_dir=movie_output_dir
            )

            data_points.append(data_point)

            # Log summary
            logging.info(f"  Input:  {movie}")
            logging.info(f"  Output: {data_point.final_parody}")
            logging.info(f"  Avg phonetic score: {data_point.average_phonetic_score:.2f}")
            logging.info(f"  All scores valid: {data_point.all_phonetic_scores_valid}")
            logging.info(f"  Tool calls made: {len(data_point.tool_calls)}")
            logging.info(f"  Attempts: {len(data_point.attempts)}")

        except Exception as e:
            logging.error(f"Error processing '{movie}': {e}")
            # Create error data point for tracking
            error_point = RLVRDataPoint(
                input_title=movie,
                model_name=model_name,
                final_parody=f"ERROR: {str(e)}",
                quality_label="error"
            )
            data_points.append(error_point)

    # Save all outputs
    logging.info(f"\n{'='*80}")
    logging.info("Saving RLVR datasets...")
    logging.info(f"{'='*80}")

    saved_files = save_rlvr_dataset(data_points, f"{output_dir}/datasets", template)

    # Print final summary
    logging.info(f"\n{'='*80}")
    logging.info("TEST SUMMARY")
    logging.info(f"{'='*80}")

    successful = [dp for dp in data_points if dp.quality_label != "error"]
    valid_phonetics = [dp for dp in successful if dp.all_phonetic_scores_valid]

    logging.info(f"Total movies tested: {len(movies)}")
    logging.info(f"Successful generations: {len(successful)}")
    logging.info(f"Valid phonetic scores: {len(valid_phonetics)}")
    logging.info(f"Total tool calls captured: {sum(len(dp.tool_calls) for dp in data_points)}")
    logging.info(f"Total reasoning traces: {sum(1 for dp in data_points if dp.thinking_trace)}")

    logging.info(f"\nOutput files:")
    for name, path in saved_files.items():
        logging.info(f"  {name}: {path}")

    logging.info(f"\n{'='*80}")
    logging.info("Test complete! Data ready for RLVR training pipeline.")
    logging.info(f"{'='*80}\n")

    return data_points


def main():
    parser = argparse.ArgumentParser(
        description='Test parody generation with popular movies and capture RLVR training data'
    )
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of movies to test (default: all 10)')
    parser.add_argument('--model', type=str, default='qwen-3-32b',
                        help='Cerebras model to use (default: qwen-3-32b)')
    parser.add_argument('--output-dir', type=str, default='./rlvr_output',
                        help='Output directory for RLVR data (default: ./rlvr_output)')
    parser.add_argument('--list-movies', action='store_true',
                        help='List the 10 popular movies and exit')

    # Template configuration arguments
    parser.add_argument('--template', type=str, default='default',
                        choices=['default', 'deepseek'],
                        help='Template preset to use (default: default)')
    parser.add_argument('--reasoning-start', type=str, default=None,
                        help='Custom reasoning start tag (e.g., "<start_working_out>")')
    parser.add_argument('--reasoning-end', type=str, default=None,
                        help='Custom reasoning end tag (e.g., "<end_working_out>")')
    parser.add_argument('--solution-start', type=str, default=None,
                        help='Custom solution start tag (e.g., "<SOLUTION>")')
    parser.add_argument('--solution-end', type=str, default=None,
                        help='Custom solution end tag (e.g., "</SOLUTION>")')
    parser.add_argument('--show-templates', action='store_true',
                        help='Show available templates and exit')

    args = parser.parse_args()

    # Show available templates
    if args.show_templates:
        print("\nAvailable RLVR Templates:")
        print("=" * 60)
        for name, tmpl in TEMPLATES.items():
            if tmpl is None:
                print(f"\n  {name}: (custom - set via CLI args)")
            else:
                print(f"\n  {name}:")
                print(f"    reasoning_start: {tmpl.reasoning_start}")
                print(f"    reasoning_end:   {tmpl.reasoning_end}")
                print(f"    solution_start:  {tmpl.solution_start or '(none)'}")
                print(f"    solution_end:    {tmpl.solution_end or '(none)'}")
        print("\n" + "=" * 60)
        print("\nExample usage:")
        print("  python test_popular_movies.py --template deepseek")
        print("  python test_popular_movies.py --reasoning-start '<think>' --reasoning-end '</think>'")
        print()
        return

    # Just list movies if requested
    if args.list_movies:
        print("\n10 Popular Movies for Testing:")
        print("-" * 40)
        for i, movie in enumerate(POPULAR_MOVIES, 1):
            print(f"  {i}. {movie}")
        print()
        return

    # Build template from args
    if args.reasoning_start or args.reasoning_end or args.solution_start or args.solution_end:
        # Custom template from CLI args
        base_template = TEMPLATES.get(args.template, DEFAULT_TEMPLATE)
        template = RLVRTemplateTags(
            reasoning_start=args.reasoning_start or base_template.reasoning_start,
            reasoning_end=args.reasoning_end or base_template.reasoning_end,
            solution_start=args.solution_start or base_template.solution_start,
            solution_end=args.solution_end or base_template.solution_end,
        )
        logging.info(f"Using custom template with tags:")
        logging.info(f"  reasoning: {template.reasoning_start}...{template.reasoning_end}")
        logging.info(f"  solution: {template.solution_start}...{template.solution_end}")
    else:
        # Use preset template
        template = TEMPLATES.get(args.template, DEFAULT_TEMPLATE)
        logging.info(f"Using '{args.template}' template preset")

    # Get API key
    api_key = os.environ.get("CEREBRAS_API_KEY")

    if not api_key:
        logging.error("CEREBRAS_API_KEY environment variable not set")
        print("\nERROR: Please set CEREBRAS_API_KEY environment variable")
        print("Example: export CEREBRAS_API_KEY='your-key-here'")
        sys.exit(1)

    # Run test
    run_popular_movies_test(
        limit=args.limit,
        model_name=args.model,
        api_key=api_key,
        output_dir=args.output_dir,
        template=template
    )


if __name__ == "__main__":
    main()
