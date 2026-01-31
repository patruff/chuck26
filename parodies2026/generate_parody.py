import json
import logging
import csv
import os
import re
import sys
import argparse
from typing import List, Optional
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from rich import print as rprint
from smolagents import CodeAgent, load_tool

# Import the Cerebras Cloud SDK
from cerebras.cloud.sdk import Cerebras

# Import word structures and prompts
from word_structures import (
    custom_phones,
    funny_words,
    KNOWN_FUNNY_PARODIES,
    get_example_prompt_text,
)
from system_prompt import (
    AGENT_SYSTEM_PROMPT,
    PARODY_STYLE_GUIDE,
    build_generation_prompt,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

# Load tools from Hugging Face Hub
parody_tool = load_tool("patruff/parody-suggestions", trust_remote_code=True)
word_phone_tool = load_tool("patruff/word-phone", trust_remote_code=True)

@dataclass
class ModelResponse:
    content: str

class CerebrasModel:
    """Model adapter for Cerebras API that works with SmolaAgents"""

    def __init__(self, model_name="qwen-3-32b", api_key=None):
        self.model_name = model_name
        self.api_key = api_key
        self.client = Cerebras(api_key=self.api_key)
        self.system_message = None
        logging.info(f"Initialized CerebrasModel with model: {model_name}")

    def _preprocess_content(self, content):
        """
        Preprocess content to handle template tags that SmolaAgents needs
        but might confuse Cerebras API
        """
        if not isinstance(content, str):
            return str(content)

        # Replace SmolaAgents templating tags with plain text equivalents
        content = content.replace("{{authorized_imports}}", "[AUTHORIZED IMPORTS PLACEHOLDER]")
        content = content.replace("{{managed_agents_descriptions}}", "[MANAGED AGENTS DESCRIPTIONS PLACEHOLDER]")

        return content

    def __call__(self, messages: List[dict], stop_sequences: Optional[List[str]] = None, **kwargs) -> ModelResponse:
        """Convert messages to Cerebras format and call the API"""
        try:
            # Format messages for Cerebras
            formatted_messages = []

            # Process all messages
            for msg in messages:
                if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                    # Preprocess the content for Cerebras API
                    processed_content = self._preprocess_content(msg['content'])

                    formatted_messages.append({
                        "role": msg["role"],
                        "content": processed_content
                    })
                else:
                    # Convert non-standard messages to user messages
                    formatted_messages.append({
                        "role": "user",
                        "content": self._preprocess_content(str(msg))
                    })

            # Log the formatted messages for debugging
            logging.info(f"Sending messages to Cerebras API: {formatted_messages}")

            # Set up completion parameters
            completion_params = {
                "max_tokens": kwargs.get("max_tokens", 4096),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
            }

            # Add stop sequences if provided
            if stop_sequences:
                completion_params["stop"] = stop_sequences

            # Call Cerebras API
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=formatted_messages,
                **completion_params
            )

            # Get the assistant's message content
            response_content = completion.choices[0].message.content

            # Return the response in our expected format
            return ModelResponse(content=response_content)

        except Exception as e:
            logging.error(f"Error calling Cerebras API: {str(e)}", exc_info=True)
            return ModelResponse(content=f"Error: {str(e)}")

class OutputCapture:
    """Captures output for both real-time display and saving to files"""

    def __init__(self, output_base_dir='./output'):
        """Initialize output capture with configurable output directory"""
        self.output_dir = Path('parody_output')
        self.output_dir.mkdir(exist_ok=True)

        # Use local output directory instead of Google Drive
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)

        self.step_counter = 0
        self.current_data = {}
        self.current_title = ""  # Store the original title

        # Create timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create raw output file
        self.raw_output_file = None
        logging.info(f"Initialized OutputCapture. Output directory: {self.output_base_dir}")

    def clean_text(self, text):
        """Clean extracted text by removing extra whitespace and newlines"""
        if text:
            return ' '.join(text.strip().split())
        return ""

    def sanitize_text(self, text):
        """Replace smart quotes with regular quotes to avoid syntax errors"""
        if text:
            # Replace smart quotes with regular quotes
            text = text.replace(''', "'").replace(''', "'").replace('"', '"').replace('"', '"')
        return text

    def extract_data(self, text):
        """Extract relevant data using regex patterns for DPO training"""
        # Extract the full thinking trace between <think> tags
        thinking_pattern = r'<think>(.*?)</think>'
        thinking_match = re.search(thinking_pattern, text, re.DOTALL)
        thinking_trace = thinking_match.group(1).strip() if thinking_match else ""

        # Extract attempts with correct pattern
        attempt_pattern = r'### Attempt (\d+):\s*\n\*\*"([^"]+)"\*\*'

        # Alternative pattern for cases where the format might be different
        alt_attempt_pattern = r'### Attempt (\d+):\s*\n\*\*([^*]+)\*\*'

        # Pattern for extracting reasoning
        reason_pattern = r'### Final Reasoning:(.*?)(?=###|\Z)'

        # Find matches
        attempts = re.findall(attempt_pattern, text, re.DOTALL)

        # If regular pattern doesn't work, try alternative
        if not attempts:
            attempts = re.findall(alt_attempt_pattern, text, re.DOTALL)

        reason_match = re.search(reason_pattern, text, re.DOTALL)

        # Get the top 2 attempts by number
        attempts.sort(key=lambda x: int(x[0]))

        # Get second parody from Attempt 2 or 3 if available
        parody2 = ""
        if len(attempts) > 1:
            parody2 = self.clean_text(attempts[1][1])
        elif len(attempts) > 2:
            parody2 = self.clean_text(attempts[2][1])

        return {
            'input': self.current_title,
            'parody1': self.clean_text(attempts[0][1]) if attempts else "",
            'parody2': parody2,
            'reasoning': reason_match.group(1).strip() if reason_match else "",
            'thinking_trace': thinking_trace
        }

    def get_next_file_number(self):
        existing_files = os.listdir(self.output_base_dir)
        numbers = [int(f.split('_')[0]) for f in existing_files if f.endswith('.csv') and f.split('_')[0].isdigit()]
        return max(numbers, default=0) + 1

    def generate_filename(self):
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        file_number = self.get_next_file_number()
        return f"{file_number}_{timestamp}_cerebras_{os.environ.get('CEREBRAS_MODEL', 'qwen-3-32b')}.csv"

    def export_to_csv(self):
        if self.current_data:
            filename = self.generate_filename()
            csv_file = os.path.join(self.output_base_dir, filename)

            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['input', 'parody1', 'parody2', 'reasoning', 'thinking_trace'])
                writer.writeheader()
                writer.writerow(self.current_data)

            print(f"\nData exported to: {filename}")

    def initialize_raw_output(self, title):
        """Initialize raw output file for this run"""
        self.current_title = title
        raw_filename = f"RAW_{title.replace(' ', '_')}_{self.timestamp}.txt"
        self.raw_output_file = os.path.join(self.output_base_dir, raw_filename)

        # Create the file and write a header
        with open(self.raw_output_file, 'w', encoding='utf-8') as f:
            f.write(f"=== RAW OUTPUT CAPTURE ===\n")
            f.write(f"Title: {title}\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"Model: {os.environ.get('CEREBRAS_MODEL', 'qwen-3-32b')}\n")
            f.write("=" * 50 + "\n\n")

        print(f"Raw output will be captured to: {self.raw_output_file}")
        return self.raw_output_file

    def write_to_raw_output(self, text):
        """Write text to the raw output file and print to console"""
        if self.raw_output_file and text:
            # Display in console
            print(text)

            # Write to file
            with open(self.raw_output_file, 'a', encoding='utf-8') as f:
                f.write(text + "\n")

    def callback(self, step_log):
        self.step_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Write to raw output before writing to individual files
        self.write_to_raw_output(f"\n=== STEP {self.step_counter} ({timestamp}) ===\n")

        # Debug file dump
        dump_file = self.output_dir / f'full_dump_{self.step_counter}_{timestamp}.txt'
        with open(dump_file, 'w', encoding='utf-8') as f:
            f.write(f"=== Full Step Log Dump ===\n")
            f.write(f"Step: {self.step_counter}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write("\n=== Attributes ===\n")
            for attr in dir(step_log):
                if not attr.startswith('_'):
                    try:
                        value = getattr(step_log, attr)
                        f.write(f"\n--- {attr} ---\n")
                        # Sanitize any text to avoid smart quotes
                        if isinstance(value, str):
                            value = self.sanitize_text(value)
                        f.write(str(value))
                    except Exception as e:
                        f.write(f"Error getting {attr}: {str(e)}")

        if hasattr(step_log, 'llm_output') and step_log.llm_output:
            # Sanitize the output to replace smart quotes
            sanitized_output = self.sanitize_text(step_log.llm_output)

            llm_file = self.output_dir / f'llm_output_{self.step_counter}_{timestamp}.txt'
            with open(llm_file, 'w', encoding='utf-8') as f:
                f.write(sanitized_output)

            self.current_data = self.extract_data(sanitized_output)
            self.export_to_csv()

            # Write to raw output
            self.write_to_raw_output(f"\n=== Assistant's Thinking (Step {self.step_counter}) ===\n")
            self.write_to_raw_output(sanitized_output)

        if hasattr(step_log, 'action_output') and step_log.action_output:
            action_file = self.output_dir / f'action_output_{self.step_counter}_{timestamp}.txt'
            with open(action_file, 'w', encoding='utf-8') as f:
                f.write(str(step_log.action_output))

            # Write to raw output
            self.write_to_raw_output(f"\n=== Action Output (Step {self.step_counter}) ===\n")
            self.write_to_raw_output(str(step_log.action_output))

def generate_parody(title: str, model_name="qwen-3-32b", api_key=None, output_dir='./output') -> str:
    """Generate parodies with pre-processed suggestions and phonetic verification."""
    # Initialize output capture
    output_capture = OutputCapture(output_base_dir=output_dir)
    raw_output_file = output_capture.initialize_raw_output(title)

    try:
        # Get initial suggestions
        words = title.split()
        suggestions = {}
        word_list_str = json.dumps(funny_words)

        output_capture.write_to_raw_output(f"Generating suggestions for words in '{title}'...")

        for word in words:
            result = parody_tool.forward(
                target=word,
                word_list_str=word_list_str,
                min_similarity="0.6",
                custom_phones=custom_phones
            )
            suggestions[word] = json.loads(result)
            output_capture.write_to_raw_output(f"Suggestions for '{word}': {len(json.loads(result))} options")

        # Use centralized system prompt from system_prompt.py
        system_prompt = AGENT_SYSTEM_PROMPT

        # Build the examples section from known funny parodies
        examples_text = get_example_prompt_text()

        # Build the generation prompt using the centralized template
        prompt = build_generation_prompt(
            title=title,
            examples_text=examples_text,
            suggestions_json=json.dumps(suggestions, indent=2)
        )

        output_capture.write_to_raw_output("\nInitializing Cerebras model...")

        # Initialize model with Cerebras
        model_wrapper = CerebrasModel(model_name=model_name, api_key=api_key)

        output_capture.write_to_raw_output("\nCreating CodeAgent with tools...")

        # Create agent with the model and tools
        agent = CodeAgent(
            tools=[word_phone_tool],
            model=model_wrapper,
            system_prompt=system_prompt,
            additional_authorized_imports=["json", "smolagents", "load_tool"],
            step_callbacks=[output_capture.callback]
        )

        output_capture.write_to_raw_output("\nRunning agent to generate parodies...")

        # Run the agent to generate parodies
        result = agent.run(prompt)

        # Write the final result to the raw output
        output_capture.write_to_raw_output("\n=== FINAL RESULT ===\n")
        output_capture.write_to_raw_output(result)

        # Extract information from the raw output
        # Extract original title
        original_title = title

        # Extract final chosen parody
        parody_sections = []
        parody_pattern = r"### Final Chosen Parody:.*?\n\*\*\"(.*?)\"\*\*"
        parody_matches = re.finditer(parody_pattern, result, re.DOTALL)

        for match in parody_matches:
            parody = match.group(1).strip()
            # Skip examples with brackets
            if '[' not in parody and ']' not in parody:
                parody_sections.append(parody)

        final_parody = parody_sections[-1] if parody_sections else "Unknown"

        # Extract reasoning
        reasoning_pattern = r"<think>(.*?)--- observations ---"
        reasoning_match = re.search(reasoning_pattern, result, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "Unknown"
        reasoning = reasoning.replace("</think>", "")

        # Create a CSV file for the extracted data
        csv_file = os.path.join(output_capture.output_base_dir, f"PARODY_{title.replace(' ', '_')}_{output_capture.timestamp}.csv")

        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Original Title", "Final Parody", "Reasoning"])
            writer.writerow([original_title, final_parody, reasoning])

        output_capture.write_to_raw_output(f"\nExtracted data saved to: {csv_file}")

        return result

    except Exception as e:
        error_msg = f"Error in parody generation: {str(e)}"
        logging.error(error_msg, exc_info=True)
        output_capture.write_to_raw_output(f"\n{error_msg}")
        return f"Error: {str(e)}"

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate parody titles using Cerebras AI')
    parser.add_argument('--title', type=str, default="The Running Man",
                        help='Title to generate parodies for')
    parser.add_argument('--model', type=str, default="qwen-3-32b",
                        help='Cerebras model to use')
    parser.add_argument('--output-dir', type=str, default='./output',
                        help='Output directory for results')

    args = parser.parse_args()

    # API key - either use the one in the code or set as environment variable
    api_key = os.environ.get("CEREBRAS_API_KEY")

    if not api_key:
        logging.error("CEREBRAS_API_KEY environment variable not set")
        print("ERROR: Please set CEREBRAS_API_KEY environment variable")
        sys.exit(1)

    print(f"\n{'='*50}")
    print(f"Generating parodies for: '{args.title}'")
    print(f"Using model: {args.model}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*50}\n")

    try:
        # Call generate_parody with the title (using funny_words from import)
        result = generate_parody(
            args.title,
            model_name=args.model,
            api_key=api_key,
            output_dir=args.output_dir
        )

        if result:
            print("\nGeneration complete.")
            print(f"- Check 'parody_output' directory for debug files")
            print(f"- Check '{args.output_dir}' directory for CSV results and raw output")
        else:
            print("\nGeneration failed - check debug.log for details")

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}", exc_info=True)
        print(f"\nExecution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
