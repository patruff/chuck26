#!/usr/bin/env python3
"""
Batch Parody Generator

Reads titles from input.csv and generates parodies for each,
writing results to output.csv at the root level.
"""

import os
import sys
import csv
import re
import argparse
import logging
from pathlib import Path
from typing import Dict, List

# Import the generate_parody function from the main script
from generate_parody import generate_parody

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def extract_parody_from_result(result: str, original_title: str) -> Dict[str, str]:
    """Extract final parody and reasoning from generation result."""

    # Extract final chosen parody
    parody_pattern = r"### Final Chosen Parody:.*?\n\*\*\"?([^\*\"]+)\"?\*\*"
    parody_match = re.search(parody_pattern, result, re.DOTALL)

    if parody_match:
        final_parody = parody_match.group(1).strip()
        # Skip examples with brackets
        if '[' in final_parody or ']' in final_parody:
            final_parody = "Generation failed - no valid parody found"
    else:
        final_parody = "Generation failed - no parody found"

    # Extract reasoning (everything between </think> and --- observations ---)
    reasoning_pattern = r"</think>(.*?)(?:--- observations ---|$)"
    reasoning_match = re.search(reasoning_pattern, result, re.DOTALL)

    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
        # Clean up the reasoning
        reasoning = reasoning.replace("</think>", "").strip()
        # Limit reasoning length for CSV
        if len(reasoning) > 500:
            reasoning = reasoning[:497] + "..."
    else:
        reasoning = "No reasoning extracted"

    return {
        'input': original_title,
        'parody_result': final_parody,
        'reasoning': reasoning
    }


def process_batch(input_csv: str, output_csv: str, model_name: str, api_key: str):
    """Process all titles from input CSV and write results to output CSV."""

    input_path = Path(input_csv)
    output_path = Path(output_csv)

    if not input_path.exists():
        logging.error(f"Input file not found: {input_csv}")
        sys.exit(1)

    # Read input titles
    titles = []
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'title' in row:
                titles.append(row['title'])
            else:
                logging.warning(f"Skipping row without 'title' column: {row}")

    if not titles:
        logging.error("No titles found in input CSV")
        sys.exit(1)

    logging.info(f"Found {len(titles)} titles to process")

    # Process each title
    results = []
    for i, title in enumerate(titles, 1):
        logging.info(f"\n{'='*80}")
        logging.info(f"Processing {i}/{len(titles)}: {title}")
        logging.info(f"{'='*80}")

        try:
            # Generate parody
            result = generate_parody(
                title=title,
                model_name=model_name,
                api_key=api_key,
                output_dir=f'./batch_output/{i:02d}_{title.replace(" ", "_")}'
            )

            # Extract parody and reasoning
            extracted = extract_parody_from_result(result, title)
            results.append(extracted)

            logging.info(f"✅ Completed: {title} → {extracted['parody_result']}")

        except Exception as e:
            logging.error(f"❌ Error processing '{title}': {e}")
            results.append({
                'input': title,
                'parody_result': f"ERROR: {str(e)}",
                'reasoning': "Generation failed due to error"
            })

    # Write results to output CSV
    logging.info(f"\n{'='*80}")
    logging.info(f"Writing results to {output_csv}")

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['input', 'parody_result', 'reasoning'])
        writer.writeheader()
        writer.writerows(results)

    logging.info(f"✅ Successfully wrote {len(results)} results to {output_csv}")
    logging.info(f"{'='*80}\n")

    # Print summary
    success_count = sum(1 for r in results if not r['parody_result'].startswith('ERROR'))
    logging.info(f"📊 BATCH SUMMARY")
    logging.info(f"Total processed: {len(results)}")
    logging.info(f"Successful: {success_count}")
    logging.info(f"Failed: {len(results) - success_count}")


def main():
    parser = argparse.ArgumentParser(description='Batch generate parodies from CSV')
    parser.add_argument('--input', type=str, default='input.csv',
                        help='Input CSV file with titles (default: input.csv)')
    parser.add_argument('--output', type=str, default='output.csv',
                        help='Output CSV file for results (default: output.csv)')
    parser.add_argument('--model', type=str, default='qwen-3-32b',
                        help='Cerebras model to use (default: qwen-3-32b)')

    args = parser.parse_args()

    # Get API key from environment
    api_key = os.environ.get("CEREBRAS_API_KEY")

    if not api_key:
        logging.error("CEREBRAS_API_KEY environment variable not set")
        sys.exit(1)

    # Process batch
    process_batch(
        input_csv=args.input,
        output_csv=args.output,
        model_name=args.model,
        api_key=api_key
    )


if __name__ == "__main__":
    main()
