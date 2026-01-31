#!/usr/bin/env python3
"""
RLVR Dataset Tools - Convert and prepare parody data for RLVR training.

This module provides tools for:
1. Labeling parody data points as "good" or "bad"
2. Converting to different training formats (SFT, DPO, RLVR)
3. Filtering and validating datasets
4. Creating training-ready datasets

Usage:
    # Label data interactively
    python rlvr_dataset_tools.py label --input data.jsonl --output labeled.jsonl

    # Convert to training format
    python rlvr_dataset_tools.py convert --input labeled.jsonl --format dpo --output training.jsonl

    # Auto-label based on quality signals
    python rlvr_dataset_tools.py auto-label --input data.jsonl --output labeled.jsonl
"""

import os
import sys
import json
import csv
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# =============================================================================
# RLVR Template Configuration
# =============================================================================
# Customizable tags for different training setups

@dataclass
class RLVRTemplateTags:
    """Configurable tags for RLVR training format."""
    reasoning_start: str = "<start_working_out>"
    reasoning_end: str = "<end_working_out>"
    solution_start: str = "<SOLUTION>"
    solution_end: str = "</SOLUTION>"

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
        if self.solution_start:
            return f"{self.solution_start}{solution_text}{self.solution_end}"
        return solution_text

    def format_full_response(self, reasoning: str, solution: str) -> str:
        """Format a complete response with reasoning and solution."""
        response = f"{self.reasoning_start}\n{reasoning}\n{self.reasoning_end}\n\n"
        if self.solution_start:
            response += f"{self.solution_start}{solution}{self.solution_end}"
        else:
            response += solution
        return response


# Preset templates
DEFAULT_TEMPLATE = RLVRTemplateTags()

DEEPSEEK_TEMPLATE = RLVRTemplateTags(
    reasoning_start="<think>",
    reasoning_end="</think>",
    solution_start="",
    solution_end=""
)

TEMPLATES = {
    "default": DEFAULT_TEMPLATE,
    "deepseek": DEEPSEEK_TEMPLATE,
}


@dataclass
class QualityCriteria:
    """Criteria for auto-labeling data points as good/bad."""
    min_phonetic_score: float = 0.6
    min_humor_rating: int = 6
    require_all_scores_valid: bool = True
    min_tool_calls: int = 2
    require_reasoning: bool = True


def load_jsonl(filepath: str) -> List[Dict]:
    """Load data from JSONL file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], filepath: str):
    """Save data to JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def auto_label_data_point(
    data_point: Dict,
    criteria: QualityCriteria
) -> Tuple[str, str]:
    """
    Auto-label a data point based on quality criteria.

    Returns:
        Tuple of (label, reason)
    """
    reasons = []

    # Check phonetic scores
    avg_score = data_point.get('average_phonetic_score', 0)
    if avg_score < criteria.min_phonetic_score:
        reasons.append(f"Low phonetic score: {avg_score:.2f} < {criteria.min_phonetic_score}")

    # Check if all scores are valid
    if criteria.require_all_scores_valid:
        if not data_point.get('all_phonetic_scores_valid', False):
            reasons.append("Not all phonetic scores > 0.6")

    # Check humor rating
    humor = data_point.get('humor_rating')
    if humor and humor < criteria.min_humor_rating:
        reasons.append(f"Low humor rating: {humor} < {criteria.min_humor_rating}")

    # Check tool calls
    tool_calls = data_point.get('tool_calls', [])
    if len(tool_calls) < criteria.min_tool_calls:
        reasons.append(f"Few tool calls: {len(tool_calls)} < {criteria.min_tool_calls}")

    # Check reasoning
    if criteria.require_reasoning:
        if not data_point.get('thinking_trace'):
            reasons.append("No reasoning trace found")

    # Check for errors
    final_parody = data_point.get('final_parody', '')
    if final_parody.startswith('ERROR') or not final_parody:
        reasons.append("Generation error or empty output")

    # Determine label
    if reasons:
        return 'bad', '; '.join(reasons)
    else:
        return 'good', 'Meets all quality criteria'


def convert_to_sft_format(data_point: Dict, template: RLVRTemplateTags = None) -> Dict:
    """
    Convert a data point to Supervised Fine-Tuning format.

    SFT format:
    {
        "instruction": "Create a funny parody of ...",
        "input": "",
        "output": "The reasoning and final answer"
    }
    """
    template = template or DEFAULT_TEMPLATE
    input_title = data_point.get('input_title', '')
    thinking = data_point.get('thinking_trace', '')
    final_parody = data_point.get('final_parody', '')
    reasoning = data_point.get('final_reasoning', '')

    # Build the full response with reasoning using template tags
    output_parts = []
    if thinking:
        output_parts.append(template.format_reasoning(thinking))
    if final_parody:
        if template.solution_start:
            output_parts.append(template.format_solution(final_parody))
        else:
            output_parts.append(f"\nFinal Parody: \"{final_parody}\"")
    if reasoning:
        output_parts.append(f"\nReasoning: {reasoning}")

    return {
        "system_prompt": template.get_system_prompt(),
        "instruction": f"Create a funny parody of the movie title '{input_title}'. "
                      f"Use phonetic similarity checking to ensure words sound similar. "
                      f"Show your reasoning process.",
        "input": "",
        "output": '\n'.join(output_parts) if output_parts else final_parody
    }


def convert_to_dpo_format(good_point: Dict, bad_point: Dict, template: RLVRTemplateTags = None) -> Dict:
    """
    Convert a pair of good/bad data points to DPO format.

    DPO format:
    {
        "prompt": "The instruction",
        "chosen": "The good response",
        "rejected": "The bad response"
    }
    """
    template = template or DEFAULT_TEMPLATE
    # Both should have the same input ideally, but we handle different inputs too
    input_title = good_point.get('input_title', bad_point.get('input_title', ''))

    good_sft = convert_to_sft_format(good_point, template)
    bad_sft = convert_to_sft_format(bad_point, template)

    return {
        "system_prompt": template.get_system_prompt(),
        "prompt": f"Create a funny parody of the movie title '{input_title}'. "
                 f"Use phonetic similarity checking to ensure words sound similar. "
                 f"Show your reasoning process.",
        "chosen": good_sft["output"],
        "rejected": bad_sft["output"]
    }


def convert_to_rlvr_format(data_point: Dict, template: RLVRTemplateTags = None) -> Dict:
    """
    Convert a data point to RLVR (Reinforcement Learning from Verifiable Rewards) format.

    RLVR format includes:
    - The prompt/instruction
    - The model's response with tool calls
    - Verifiable reward signals (phonetic scores, etc.)
    """
    template = template or DEFAULT_TEMPLATE
    input_title = data_point.get('input_title', '')
    tool_calls = data_point.get('tool_calls', [])
    thinking = data_point.get('thinking_trace', '')
    final_parody = data_point.get('final_parody', '')

    # Build tool call trace
    tool_trace = []
    for tc in tool_calls:
        tool_trace.append({
            "tool": tc.get('tool_name', 'word_phone_tool'),
            "args": tc.get('arguments', {}),
            "result": tc.get('result')
        })

    # Format tool calls as part of reasoning
    tool_calls_text = ""
    if tool_calls:
        tool_calls_text = "\n\nTool calls made:\n"
        for tc in tool_calls:
            args = tc.get('arguments', {})
            tool_calls_text += f"- {tc.get('tool_name', 'word_phone_tool')}(\"{args.get('word1', '')}\", \"{args.get('word2', '')}\") → {tc.get('result')}\n"

    full_reasoning = thinking + tool_calls_text

    return {
        "system_prompt": template.get_system_prompt(),
        "prompt": f"Create a funny parody of the movie title '{input_title}'.",
        "response": template.format_full_response(full_reasoning, final_parody),
        "response_structured": {
            "thinking": thinking,
            "tool_calls": tool_trace,
            "final_answer": final_parody
        },
        "template": {
            "reasoning_start": template.reasoning_start,
            "reasoning_end": template.reasoning_end,
            "solution_start": template.solution_start,
            "solution_end": template.solution_end
        },
        "rewards": {
            "phonetic_validity": data_point.get('all_phonetic_scores_valid', False),
            "average_phonetic_score": data_point.get('average_phonetic_score', 0),
            "humor_rating": data_point.get('humor_rating'),
            "tool_usage_count": len(tool_calls),
            "has_reasoning": bool(thinking)
        },
        "verifiable_checks": [
            {
                "type": "phonetic_score",
                "word1": tc.get('arguments', {}).get('word1', ''),
                "word2": tc.get('arguments', {}).get('word2', ''),
                "score": tc.get('result'),
                "passed": tc.get('result', 0) > 0.6 if tc.get('result') else False
            }
            for tc in tool_calls
        ]
    }


def convert_to_tool_use_format(data_point: Dict) -> List[Dict]:
    """
    Convert a data point to tool-use training format.
    Creates one training example per tool call.
    """
    input_title = data_point.get('input_title', '')
    tool_calls = data_point.get('tool_calls', [])
    thinking = data_point.get('thinking_trace', '')

    examples = []
    for tc in tool_calls:
        examples.append({
            "context": f"You are creating a parody of '{input_title}'. "
                      f"You need to check if two words sound similar.",
            "thought": f"I should check if '{tc.get('arguments', {}).get('word1', '')}' "
                      f"sounds similar to '{tc.get('arguments', {}).get('word2', '')}'",
            "tool_call": {
                "name": tc.get('tool_name', 'word_phone_tool'),
                "arguments": tc.get('arguments', {})
            },
            "tool_result": tc.get('result'),
            "interpretation": f"The phonetic similarity score is {tc.get('result')}. "
                            f"{'This is acceptable (> 0.6).' if tc.get('result', 0) > 0.6 else 'This is too low (< 0.6), need to try another word.'}"
        })

    return examples


def create_dpo_pairs(data: List[Dict], template: RLVRTemplateTags = None) -> List[Dict]:
    """
    Create DPO pairs from labeled data.
    Pairs good examples with bad examples.
    """
    template = template or DEFAULT_TEMPLATE
    good_examples = [d for d in data if d.get('quality_label') == 'good']
    bad_examples = [d for d in data if d.get('quality_label') == 'bad']

    if not good_examples or not bad_examples:
        logging.warning("Need both good and bad examples for DPO pairs")
        return []

    pairs = []

    # Create pairs (each good example paired with each bad example)
    for good in good_examples:
        for bad in bad_examples:
            pairs.append(convert_to_dpo_format(good, bad, template))

    return pairs


def interactive_labeler(data: List[Dict]) -> List[Dict]:
    """
    Interactively label data points.
    """
    labeled_data = []

    print("\n" + "="*60)
    print("INTERACTIVE DATA LABELER")
    print("="*60)
    print("\nCommands:")
    print("  g - Label as 'good'")
    print("  b - Label as 'bad'")
    print("  s - Skip this example")
    print("  q - Quit and save progress")
    print("="*60 + "\n")

    for i, item in enumerate(data):
        print(f"\n[{i+1}/{len(data)}] {'='*50}")
        print(f"Input Title: {item.get('input_title', 'N/A')}")
        print(f"Final Parody: {item.get('final_parody', 'N/A')}")
        print(f"Avg Phonetic Score: {item.get('average_phonetic_score', 0):.2f}")
        print(f"All Scores Valid: {item.get('all_phonetic_scores_valid', False)}")
        print(f"Humor Rating: {item.get('humor_rating', 'N/A')}")
        print(f"Tool Calls: {len(item.get('tool_calls', []))}")

        if item.get('thinking_trace'):
            # Show truncated thinking trace
            trace = item['thinking_trace'][:200] + "..." if len(item.get('thinking_trace', '')) > 200 else item['thinking_trace']
            print(f"Reasoning (truncated): {trace}")

        print("-"*50)

        while True:
            choice = input("Label (g=good, b=bad, s=skip, q=quit): ").strip().lower()

            if choice == 'g':
                item['quality_label'] = 'good'
                item['label_source'] = 'manual'
                labeled_data.append(item)
                print("  -> Labeled as GOOD")
                break
            elif choice == 'b':
                item['quality_label'] = 'bad'
                item['label_source'] = 'manual'
                labeled_data.append(item)
                print("  -> Labeled as BAD")
                break
            elif choice == 's':
                print("  -> Skipped")
                break
            elif choice == 'q':
                print("\nSaving progress and exiting...")
                return labeled_data
            else:
                print("Invalid choice. Use g, b, s, or q.")

    return labeled_data


def auto_label_dataset(
    data: List[Dict],
    criteria: Optional[QualityCriteria] = None
) -> List[Dict]:
    """
    Auto-label all data points based on quality criteria.
    """
    if criteria is None:
        criteria = QualityCriteria()

    labeled_data = []

    for item in data:
        label, reason = auto_label_data_point(item, criteria)
        item['quality_label'] = label
        item['label_source'] = 'auto'
        item['label_reason'] = reason
        labeled_data.append(item)

    # Print summary
    good_count = sum(1 for d in labeled_data if d['quality_label'] == 'good')
    bad_count = sum(1 for d in labeled_data if d['quality_label'] == 'bad')

    logging.info(f"Auto-labeling complete:")
    logging.info(f"  Good: {good_count}")
    logging.info(f"  Bad: {bad_count}")

    return labeled_data


def convert_dataset(
    data: List[Dict],
    output_format: str,
    template: RLVRTemplateTags = None
) -> List[Dict]:
    """
    Convert dataset to specified training format.

    Formats:
    - sft: Supervised Fine-Tuning format
    - dpo: Direct Preference Optimization format (requires labeled data)
    - rlvr: RLVR format with verifiable rewards
    - tool: Tool-use training format
    """
    template = template or DEFAULT_TEMPLATE

    if output_format == 'sft':
        return [convert_to_sft_format(d, template) for d in data if d.get('quality_label') == 'good']

    elif output_format == 'dpo':
        return create_dpo_pairs(data, template)

    elif output_format == 'rlvr':
        return [convert_to_rlvr_format(d, template) for d in data]

    elif output_format == 'tool':
        all_examples = []
        for d in data:
            all_examples.extend(convert_to_tool_use_format(d))
        return all_examples

    else:
        raise ValueError(f"Unknown format: {output_format}")


def print_dataset_stats(data: List[Dict]):
    """Print statistics about the dataset."""
    total = len(data)
    good = sum(1 for d in data if d.get('quality_label') == 'good')
    bad = sum(1 for d in data if d.get('quality_label') == 'bad')
    unlabeled = total - good - bad

    avg_phonetic = sum(d.get('average_phonetic_score', 0) for d in data) / total if total else 0
    total_tool_calls = sum(len(d.get('tool_calls', [])) for d in data)
    with_reasoning = sum(1 for d in data if d.get('thinking_trace'))

    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    print(f"Total examples: {total}")
    print(f"  - Good: {good}")
    print(f"  - Bad: {bad}")
    print(f"  - Unlabeled: {unlabeled}")
    print(f"\nQuality Metrics:")
    print(f"  - Average phonetic score: {avg_phonetic:.2f}")
    print(f"  - Total tool calls: {total_tool_calls}")
    print(f"  - Examples with reasoning: {with_reasoning}")
    print("="*50 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='RLVR Dataset Tools - Prepare parody data for training'
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Label command
    label_parser = subparsers.add_parser('label', help='Interactively label data')
    label_parser.add_argument('--input', required=True, help='Input JSONL file')
    label_parser.add_argument('--output', required=True, help='Output JSONL file')

    # Auto-label command
    auto_parser = subparsers.add_parser('auto-label', help='Auto-label data based on quality')
    auto_parser.add_argument('--input', required=True, help='Input JSONL file')
    auto_parser.add_argument('--output', required=True, help='Output JSONL file')
    auto_parser.add_argument('--min-phonetic', type=float, default=0.6,
                            help='Minimum phonetic score (default: 0.6)')
    auto_parser.add_argument('--min-humor', type=int, default=6,
                            help='Minimum humor rating (default: 6)')

    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert to training format')
    convert_parser.add_argument('--input', required=True, help='Input JSONL file')
    convert_parser.add_argument('--output', required=True, help='Output JSONL file')
    convert_parser.add_argument('--format', required=True,
                               choices=['sft', 'dpo', 'rlvr', 'tool'],
                               help='Output format')
    convert_parser.add_argument('--template', type=str, default='default',
                               choices=['default', 'deepseek'],
                               help='Template preset (default: default)')
    convert_parser.add_argument('--reasoning-start', type=str, default=None,
                               help='Custom reasoning start tag')
    convert_parser.add_argument('--reasoning-end', type=str, default=None,
                               help='Custom reasoning end tag')
    convert_parser.add_argument('--solution-start', type=str, default=None,
                               help='Custom solution start tag')
    convert_parser.add_argument('--solution-end', type=str, default=None,
                               help='Custom solution end tag')

    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show dataset statistics')
    stats_parser.add_argument('--input', required=True, help='Input JSONL file')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == 'label':
        data = load_jsonl(args.input)
        labeled = interactive_labeler(data)
        save_jsonl(labeled, args.output)
        logging.info(f"Saved {len(labeled)} labeled examples to {args.output}")

    elif args.command == 'auto-label':
        data = load_jsonl(args.input)
        criteria = QualityCriteria(
            min_phonetic_score=args.min_phonetic,
            min_humor_rating=args.min_humor
        )
        labeled = auto_label_dataset(data, criteria)
        save_jsonl(labeled, args.output)
        logging.info(f"Saved {len(labeled)} auto-labeled examples to {args.output}")

    elif args.command == 'convert':
        data = load_jsonl(args.input)

        # Build template from args
        if args.reasoning_start or args.reasoning_end or args.solution_start or args.solution_end:
            base_template = TEMPLATES.get(args.template, DEFAULT_TEMPLATE)
            template = RLVRTemplateTags(
                reasoning_start=args.reasoning_start or base_template.reasoning_start,
                reasoning_end=args.reasoning_end or base_template.reasoning_end,
                solution_start=args.solution_start if args.solution_start is not None else base_template.solution_start,
                solution_end=args.solution_end if args.solution_end is not None else base_template.solution_end,
            )
            logging.info(f"Using custom template: {template.reasoning_start}...{template.reasoning_end}")
        else:
            template = TEMPLATES.get(args.template, DEFAULT_TEMPLATE)
            logging.info(f"Using '{args.template}' template preset")

        converted = convert_dataset(data, args.format, template)
        save_jsonl(converted, args.output)
        logging.info(f"Converted {len(data)} examples to {len(converted)} {args.format} format examples")
        logging.info(f"Saved to {args.output}")

    elif args.command == 'stats':
        data = load_jsonl(args.input)
        print_dataset_stats(data)


if __name__ == "__main__":
    main()
