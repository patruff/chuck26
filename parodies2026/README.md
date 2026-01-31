# Parody Generator

A tool for generating creative parody titles using Cerebras AI, [smolagents](https://github.com/huggingface/smolagents), and phonetic similarity algorithms. Built for RLVR (Reinforcement Learning from Verifiable Rewards) dataset generation.

## Architecture

This project uses **smolagents** (Hugging Face's agent framework) to orchestrate AI-powered parody generation with tool use:

```
┌─────────────────────────────────────────────────────────────┐
│                     smolagents CodeAgent                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Cerebras LLM (qwen-3-32b)              │   │
│  │         Generates creative parodies with            │   │
│  │            structured reasoning (<think>)           │   │
│  └─────────────────────────────────────────────────────┘   │
│                            │                                 │
│                     Tool Calls                              │
│                            ▼                                 │
│  ┌──────────────────┐  ┌──────────────────────────────┐   │
│  │ parody_tool      │  │ word_phone_tool               │   │
│  │ (suggestions)    │  │ (phonetic verification)       │   │
│  │ patruff/parody-  │  │ patruff/word-phone            │   │
│  │ suggestions      │  │                               │   │
│  └──────────────────┘  └──────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Workflow Summary

1. **📥 Input**: Upload CSV files with titles to `parodiesdata/input/` in Google Drive
2. **⚙️ Processing**: GitHub Actions runs every 6 hours, processes new files
3. **📤 Output**: Results saved to `parodiesdata/output/` with parodies + reasoning
4. **🎯 DPO**: Manually curate chosen vs rejected parodies in `parodiesdata/dpo/`

## Features

- **smolagents integration** - Uses Hugging Face's agent framework for tool orchestration
- Generates parody titles with phonetic similarity verification
- Uses Cerebras Cloud SDK for fast AI inference
- Custom phonetic pronunciations via CMU dictionary
- **100 known parody examples** for few-shot learning (`known100.csv`)
- **RLVR training data capture** with reasoning traces and tool calls
- Google Drive data management with organized folders
- Automated batch processing via GitHub Actions
- Configurable chat templates for base model training

## Project Structure

```
parodies2026/
├── generate_parody.py      # Main parody generation script
├── system_prompt.py        # Centralized prompts (easy to edit)
├── word_structures.py      # Funny words, custom pronunciations
├── known100.csv            # 100 known funny parodies (training data)
├── test_popular_movies.py  # RLVR test with 10 popular movies
├── rlvr_dataset_tools.py   # Dataset labeling & format conversion
├── batch_generate.py       # Batch processing from CSV
└── README.md
```

### Key Files

| File | Purpose |
|------|---------|
| `system_prompt.py` | All prompts in one place - edit to customize agent behavior |
| `known100.csv` | 100 verified parodies with reasoning - add more examples here |
| `word_structures.py` | Funny word categories, custom phonetics for non-dictionary words |
| `generate_parody.py` | Main generation logic using smolagents + Cerebras |
| `test_popular_movies.py` | Generate RLVR training data from 10 popular movies |
| `rlvr_dataset_tools.py` | Convert captured data to SFT/DPO/RLVR formats |

## smolagents & Tools

This project uses [smolagents](https://github.com/huggingface/smolagents) (Hugging Face's lightweight agent framework) to enable the LLM to use tools during parody generation.

### How smolagents Works

smolagents provides a `CodeAgent` that:
1. Receives a prompt with available tools
2. Generates Python code to call those tools
3. Executes the code and observes results
4. Iterates until task completion

```python
from smolagents import CodeAgent, load_tool

# Load tools from Hugging Face Hub
parody_tool = load_tool("patruff/parody-suggestions", trust_remote_code=True)
word_phone_tool = load_tool("patruff/word-phone", trust_remote_code=True)

# Create agent with tools
agent = CodeAgent(
    tools=[word_phone_tool],
    model=cerebras_model,
    system_prompt=AGENT_SYSTEM_PROMPT,
)

# Run agent
result = agent.run("Create a parody for 'The Matrix'")
```

### Custom Tools (Hugging Face Hub)

| Tool | Hub ID | Purpose |
|------|--------|---------|
| **parody-suggestions** | `patruff/parody-suggestions` | Finds phonetically similar funny words |
| **word-phone** | `patruff/word-phone` | Calculates phonetic similarity score (0.0-1.0) |

#### parody-suggestions

Returns candidate replacement words with similarity scores:

```python
parody_tool.forward(
    target="Matrix",
    word_list_str='["mattress", "madness", "maxis"]',
    min_similarity="0.6",
    custom_phones={"serbed": {"primary_phones": "S ER B D"}}
)
# → [{"word": "Mattress", "score": 0.82, "reason": "Strong -atrix/-attress match"}]
```

#### word-phone

Verifies phonetic similarity between two words:

```python
word_phone_tool.forward("Matrix", "Mattress")
# → 0.82  (scores > 0.6 are acceptable)
```

### Phonetic Similarity Algorithm

Both tools use the CMU Pronouncing Dictionary to:
1. Convert words to phoneme sequences (e.g., "Matrix" → "M AE T R IH K S")
2. Calculate Levenshtein distance between phoneme sequences
3. Normalize to 0.0-1.0 score (higher = more similar)

**Threshold**: Score > 0.6 means the parody sounds similar enough to the original.

## Example Results

Here are 3 detailed examples showing the full generation pipeline with tool calls and reasoning:

---

### Example 1: "The Matrix"

**Input:** `The Matrix`

**Step 1 - Word Suggestions Tool Call:**
```json
parody_suggestions_tool("Matrix", word_list=funny_words)
→ [
    {"word": "Mattress", "score": 0.82, "reason": "Strong rhyme on '-atrix/-attress'"},
    {"word": "Maxis", "score": 0.71, "reason": "Similar ending pattern"},
    {"word": "Pratfall", "score": 0.63, "reason": "Matching 'at' vowel sound"}
  ]
```

**Step 2 - Phonetic Verification Tool Calls:**
```
word_phone_tool("Matrix", "Mattress") → 0.82
word_phone_tool("The", "The")        → 1.00 (unchanged)
```

**Step 3 - Model Reasoning:**
```
<think>
Looking at "The Matrix" - the key word to parody is "Matrix".

Attempt 1: "The Mattress"
- Matrix → Mattress: Strong phonetic match (0.82)
- Humor: 8/10 - Transforms sci-fi epic into something about sleeping/bedding
- The mundane "mattress" vs the profound "matrix" creates absurdist comedy

Attempt 2: "The Maxis"
- Matrix → Maxis: Score 0.71, acceptable but less funny
- Just sounds like a brand name, not inherently humorous

Selection: "The Mattress" wins - highest humor with excellent phonetics
</think>
```

**Final Output:** `The Mattress`

**Reasoning:** "Mattress" perfectly mirrors "Matrix" phonetically (0.82) while creating maximum comedic contrast - a mind-bending sci-fi becomes a furniture commercial.

---

### Example 2: "Pulp Fiction"

**Input:** `Pulp Fiction`

**Step 1 - Word Suggestions Tool Calls:**
```json
parody_suggestions_tool("Pulp", word_list=funny_words)
→ [
    {"word": "Gulp", "score": 0.85, "reason": "Perfect rhyme"},
    {"word": "Sculpt", "score": 0.72, "reason": "Similar consonant cluster"},
    {"word": "Pulpit", "score": 0.68, "reason": "Shared stem"}
  ]

parody_suggestions_tool("Fiction", word_list=funny_words)
→ [
    {"word": "Friction", "score": 0.91, "reason": "Near-perfect rhyme"},
    {"word": "Diction", "score": 0.88, "reason": "Strong rhyme match"},
    {"word": "Eviction", "score": 0.84, "reason": "Same ending pattern"}
  ]
```

**Step 2 - Phonetic Verification Tool Calls:**
```
word_phone_tool("Pulp", "Gulp")         → 0.85
word_phone_tool("Fiction", "Friction")  → 0.91
word_phone_tool("Pulp", "Sculpt")       → 0.72
word_phone_tool("Fiction", "Eviction")  → 0.84
```

**Step 3 - Model Reasoning:**
```
<think>
"Pulp Fiction" - two words to work with, lots of options!

Attempt 1: "Gulp Friction"
- Pulp → Gulp: 0.85 ✓
- Fiction → Friction: 0.91 ✓
- Humor: 9/10 - Sounds like a medical condition or awkward eating situation
- Both words changed but still sounds like original when spoken fast

Attempt 2: "Sculpt Fiction"
- Pulp → Sculpt: 0.72 ✓
- Fiction → Fiction: unchanged
- Humor: 5/10 - Just sounds like art criticism, not very funny

Attempt 3: "Pulp Eviction"
- Pulp → Pulp: unchanged
- Fiction → Eviction: 0.84 ✓
- Humor: 7/10 - About getting kicked out, some dark humor potential

Selection: "Gulp Friction" - highest phonetic scores AND funniest result
</think>
```

**Final Output:** `Gulp Friction`

**Reasoning:** Both words achieve excellent phonetic scores (0.85, 0.91) while creating an absurd phrase that sounds like either a digestive issue or an uncomfortable swallowing situation. The combination maintains the punchy rhythm of the original.

---

### Example 3: "The Godfather"

**Input:** `The Godfather`

**Step 1 - Word Suggestions Tool Call:**
```json
parody_suggestions_tool("Godfather", word_list=funny_words)
→ [
    {"word": "Oddfather", "score": 0.79, "reason": "Rhymes on '-odfather'"},
    {"word": "Rodfather", "score": 0.81, "reason": "Strong consonant match"},
    {"word": "Codfather", "score": 0.83, "reason": "Perfect rhyme pattern"},
    {"word": "Bodfather", "score": 0.77, "reason": "Similar opening sound"}
  ]
```

**Step 2 - Phonetic Verification Tool Calls:**
```
word_phone_tool("Godfather", "Codfather") → 0.83
word_phone_tool("Godfather", "Rodfather") → 0.81
word_phone_tool("Godfather", "Oddfather") → 0.79
```

**Step 3 - Model Reasoning:**
```
<think>
"The Godfather" - iconic title, the compound word "Godfather" is the target.

Attempt 1: "The Codfather"
- Godfather → Codfather: 0.83 ✓
- Humor: 9/10 - A mafia boss who runs a fish empire!
- "I'm gonna make him a flounder he can't refuse"
- Perfect blend of crime drama and seafood comedy

Attempt 2: "The Rodfather"
- Godfather → Rodfather: 0.81 ✓
- Humor: 7/10 - Could be about fishing or... other things
- Less universally funny, more niche

Attempt 3: "The Oddfather"
- Godfather → Oddfather: 0.79 ✓
- Humor: 6/10 - Just means a weird dad, too straightforward

Selection: "The Codfather" - excellent phonetics with the funniest concept
</think>
```

**Final Output:** `The Codfather`

**Reasoning:** "Codfather" scores 0.83 phonetically and creates brilliant imagery of a fish-themed crime boss. The single-letter substitution (G→C) maximizes sound similarity while completely transforming the meaning into seafood-based organized crime comedy.

---

### Quick Reference Table

| Input | Output | Phonetic Score | Key Tool Calls |
|-------|--------|----------------|----------------|
| The Matrix | The Mattress | 0.82 | `word_phone_tool("Matrix", "Mattress") → 0.82` |
| Pulp Fiction | Gulp Friction | 0.85, 0.91 | `word_phone_tool("Pulp", "Gulp") → 0.85`<br>`word_phone_tool("Fiction", "Friction") → 0.91` |
| The Godfather | The Codfather | 0.83 | `word_phone_tool("Godfather", "Codfather") → 0.83` |

*All examples show phonetic scores > 0.6 threshold, ensuring the parody sounds similar to the original.*

## Setup

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your Cerebras API key:
```bash
export CEREBRAS_API_KEY="your-api-key-here"
```

3. Run the script (single title):
```bash
python generate_parody.py --title "Your Title Here"
```

4. Or run batch processing from CSV:
```bash
python batch_generate.py --input input.csv --output output.csv
```

### Google Drive Workflow

This repository uses Google Drive for all data management with an organized folder structure.

#### Folder Structure

```
parodiesdata/
├── input/          # CSV files with titles to process
├── output/         # Generated parodies with reasoning
└── dpo/            # Human-annotated chosen vs rejected parodies
```

#### Setup

1. **Create Google Cloud Project and Service Account:**
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project
   - Enable the Google Drive API
   - Create a Service Account with Drive access
   - Generate and download a JSON key file

2. **Set up Google Drive folders:**
   - The workflow will automatically create the `parodiesdata` folder structure
   - Or manually create: `parodiesdata/input/`, `parodiesdata/output/`, `parodiesdata/dpo/`
   - Share the `parodiesdata` folder with your service account email (from JSON key)
   - Give it "Editor" permissions

3. **Add secrets to GitHub repository:**
   - Go to Settings → Secrets and variables → Actions
   - Add `CEREBRAS_API_KEY` with your Cerebras API key
   - Add `GOOGLE_DRIVE_CREDENTIALS` with the entire JSON key file contents

4. **Add input CSV files to Drive:**
   - Upload CSV files to `parodiesdata/input/` folder
   - Format: Must have a `title` column
   ```csv
   title
   The Matrix
   Die Hard
   Fight Club
   ```

#### How It Works

1. **Automated Processing:**
   - Workflow runs every 6 hours (or manually via Actions tab)
   - Checks `parodiesdata/input/` for new CSV files
   - Tracks processed files to avoid duplicates
   - Generates parodies for each title

2. **Output:**
   - Results saved to `parodiesdata/output/` as CSV files
   - Each output file includes:
     - `input`: Original title
     - `parody_result`: Generated parody
     - `reasoning`: Model's detailed reasoning
     - `timestamp`: When it was generated

3. **DPO Data Collection:**
   - Use `parodiesdata/dpo/` folder for human annotations
   - Compare model outputs and mark chosen vs rejected parodies
   - Build training data for preference learning

#### Manual Triggers

Run the workflow manually:
- Go to Actions tab
- Select "Generate Parodies from Google Drive" workflow
- Click "Run workflow"
- Select model (optional, defaults to "qwen-3-32b")

## Command Line Options

### Single Title Generation

```bash
python generate_parody.py [OPTIONS]

Options:
  --title TEXT        Title to generate parodies for (default: "The Running Man")
  --model TEXT        Cerebras model to use (default: "qwen-3-32b")
  --output-dir TEXT   Output directory for results (default: "./output")
```

### Batch CSV Processing

```bash
python batch_generate.py [OPTIONS]

Options:
  --input TEXT        Input CSV file with titles (default: "input.csv")
  --output TEXT       Output CSV file for results (default: "output.csv")
  --model TEXT        Cerebras model to use (default: "qwen-3-32b")
```

## Output

### Google Drive Output

All results are stored in Google Drive's `parodiesdata/output/` folder:

**Output CSV format:**
```csv
input,parody_result,reasoning,timestamp
The Matrix,The Mattress,"Reasoning here...",2025-01-23 14:30:00
Die Hard,Dye Lard,"Reasoning here...",2025-01-23 14:35:00
```

Each output file is named: `output_[original_filename]_[timestamp].csv`

### Local Development Output

When running scripts locally:

**Single title (`generate_parody.py`):**
- `output/`: CSV files and raw transcripts
- `parody_output/`: Debug logs
- `debug.log`: Overall execution log

**Batch processing (`batch_generate.py`):**
- `output.csv`: Results in root directory
- `batch_output/`: Per-title processing logs
- `debug.log`: Overall execution log

*Note: Local output files are gitignored. The source of truth is in Google Drive.*

## Configuration

### Custom Words

Edit `word_structures.py` to add:
- **custom_phones**: Custom phonetic pronunciations for words not in CMU dictionary
- **funny_words**: List of funny/edgy words to use as suggestions

## Requirements

- Python 3.10+
- Cerebras API key
- Dependencies listed in `requirements.txt`

## Editing Known Parodies

The `known100.csv` file contains 100 verified funny parodies used as few-shot examples:

```csv
original,parody,reasoning
The Matrix,The Mattress,Sci-fi epic becomes furniture commercial - phonetic match on -atrix/-attress
Die Hard,Dye Hard,Action movie becomes hair salon drama - single letter swap
You Got Served,You Got Serbed,Serbian ethnic humor - sounds nearly identical to original
```

**To add new examples:**
1. Open `known100.csv` in any spreadsheet or text editor
2. Add rows with: `original_title,parody_title,reasoning`
3. The system automatically loads examples on startup

## Editing Prompts

All prompts are centralized in `system_prompt.py`:

```python
# Edit the agent's core behavior
AGENT_SYSTEM_PROMPT = """Your role is to generate creative parodies..."""

# Edit the style guide
PARODY_STYLE_GUIDE = """
1. PHONETIC PRIORITY: The parody must sound like the original...
2. HUMOR THROUGH CONTRAST: Subvert expectations...
"""

# Edit the generation prompt template
GENERATION_PROMPT_TEMPLATE = """Create a funny parody of: "{title}"..."""
```

## Editing Funny Words

The `word_structures.py` file contains categorized funny words:

```python
FUNNY_WORDS_BY_CATEGORY = {
    "bodily_functions": ["fart", "poop", "butt", ...],
    "food": ["taco", "wiener", "pickle", ...],
    "animals": ["monkey", "weasel", "platypus", ...],
    # Add your own categories
}
```

**To add custom pronunciations** (for words not in CMU dictionary):

```python
custom_phones = {
    "serbed": {"primary_phones": "S ER B D"},
    "codfather": {"primary_phones": "K AA D F AA DH ER"},
    # Add phonemes using CMU format
}
```

## RLVR Training Data

This project includes tools for capturing and preparing training data for RLVR (Reinforcement Learning from Verifiable Rewards).

### Popular Movies Test

Test parody generation with 10 popular movies while capturing structured data:

```bash
# List the 10 test movies
python test_popular_movies.py --list-movies

# Run test on all 10 movies
python test_popular_movies.py

# Run test on first 3 movies only
python test_popular_movies.py --limit 3

# Custom output directory
python test_popular_movies.py --output-dir ./my_rlvr_data

# Use DeepSeek-style <think> tags
python test_popular_movies.py --template deepseek

# Use custom reasoning tags for your own chat template
python test_popular_movies.py \
    --reasoning-start "<start_working_out>" \
    --reasoning-end "<end_working_out>" \
    --solution-start "<SOLUTION>" \
    --solution-end "</SOLUTION>"

# Show available template presets
python test_popular_movies.py --show-templates
```

**Default Test Movies:**
1. The Matrix
2. Die Hard
3. Fight Club
4. Star Wars
5. Top Gun
6. The Godfather
7. Pulp Fiction
8. Forrest Gump
9. The Shining
10. Jurassic Park

**Output Files Generated:**
```
rlvr_output/
├── datasets/
│   ├── rlvr_dataset_TIMESTAMP.jsonl      # Full dataset (JSON Lines)
│   ├── rlvr_dataset_TIMESTAMP.json       # Full dataset (JSON)
│   ├── rlvr_summary_TIMESTAMP.csv        # Quick summary CSV
│   ├── tool_calls_TIMESTAMP.jsonl        # Tool calls only (for tool-use training)
│   ├── reasoning_traces_TIMESTAMP.jsonl  # Reasoning only (for reasoning training)
│   ├── rlvr_training_TIMESTAMP.jsonl     # RLVR format with configurable tags
│   └── template_config_TIMESTAMP.json    # Template configuration used
└── individual/
    └── 01_The_Matrix/                    # Per-movie debug output
```

### Dataset Labeling & Conversion

Use `rlvr_dataset_tools.py` to prepare data for training:

```bash
# Auto-label based on quality criteria (phonetic scores, humor ratings)
python rlvr_dataset_tools.py auto-label \
    --input rlvr_output/datasets/rlvr_dataset.jsonl \
    --output labeled_data.jsonl

# Interactive manual labeling
python rlvr_dataset_tools.py label \
    --input rlvr_dataset.jsonl \
    --output labeled_data.jsonl

# View dataset statistics
python rlvr_dataset_tools.py stats --input labeled_data.jsonl
```

### Converting to Training Formats

```bash
# Supervised Fine-Tuning (SFT) format
python rlvr_dataset_tools.py convert \
    --input labeled_data.jsonl \
    --format sft \
    --output sft_training.jsonl

# Direct Preference Optimization (DPO) format
python rlvr_dataset_tools.py convert \
    --input labeled_data.jsonl \
    --format dpo \
    --output dpo_training.jsonl

# RLVR format with verifiable rewards
python rlvr_dataset_tools.py convert \
    --input labeled_data.jsonl \
    --format rlvr \
    --output rlvr_training.jsonl

# Tool-use training format
python rlvr_dataset_tools.py convert \
    --input labeled_data.jsonl \
    --format tool \
    --output tool_training.jsonl

# Convert with custom template tags (for base model training)
python rlvr_dataset_tools.py convert \
    --input labeled_data.jsonl \
    --format rlvr \
    --template default \
    --output rlvr_training.jsonl

# Use DeepSeek-style tags
python rlvr_dataset_tools.py convert \
    --input labeled_data.jsonl \
    --format sft \
    --template deepseek \
    --output sft_deepseek.jsonl
```

### Chat Template Configuration

For RLVR training with base models, you can customize the reasoning tags:

**Default Template:**
```
reasoning_start: <start_working_out>
reasoning_end:   <end_working_out>
solution_start:  <SOLUTION>
solution_end:    </SOLUTION>
```

**DeepSeek Template:**
```
reasoning_start: <think>
reasoning_end:   </think>
solution_start:  (none)
solution_end:    (none)
```

**System Prompt Generated:**
```
You are given a problem.
Think about the problem and provide your working out.
Place it between <start_working_out> and <end_working_out>.
Then, provide your solution between <SOLUTION> and </SOLUTION>.
```

### Training Format Examples

**SFT Format (with default template):**
```json
{
  "system_prompt": "You are given a problem...",
  "instruction": "Create a funny parody of the movie title 'The Matrix'...",
  "input": "",
  "output": "<start_working_out>\nLooking at 'The Matrix'...\n<end_working_out>\n\n<SOLUTION>The Mattress</SOLUTION>"
}
```

**DPO Format:**
```json
{
  "system_prompt": "You are given a problem...",
  "prompt": "Create a funny parody of the movie title 'The Matrix'...",
  "chosen": "<start_working_out>Good reasoning...</end_working_out>\n\n<SOLUTION>The Mattress</SOLUTION>",
  "rejected": "<start_working_out>Poor reasoning...</end_working_out>\n\n<SOLUTION>The Maxis</SOLUTION>"
}
```

**RLVR Format (with template info):**
```json
{
  "system_prompt": "You are given a problem...",
  "prompt": "Create a funny parody of the movie title 'The Matrix'.",
  "response": "<start_working_out>\nLooking at 'The Matrix'...\n\nTool calls made:\n- word_phone_tool(\"Matrix\", \"Mattress\") → 0.82\n<end_working_out>\n\n<SOLUTION>The Mattress</SOLUTION>",
  "response_structured": {
    "thinking": "Looking at 'The Matrix' - the key word to parody is 'Matrix'...",
    "tool_calls": [
      {"tool": "word_phone_tool", "args": {"word1": "Matrix", "word2": "Mattress"}, "result": 0.82}
    ],
    "final_answer": "The Mattress"
  },
  "template": {
    "reasoning_start": "<start_working_out>",
    "reasoning_end": "<end_working_out>",
    "solution_start": "<SOLUTION>",
    "solution_end": "</SOLUTION>"
  },
  "rewards": {
    "phonetic_validity": true,
    "average_phonetic_score": 0.82,
    "humor_rating": 8,
    "tool_usage_count": 1
  },
  "verifiable_checks": [
    {"type": "phonetic_score", "word1": "Matrix", "word2": "Mattress", "score": 0.82, "passed": true}
  ]
}
```

### Quality Criteria for Auto-Labeling

Data points are labeled "good" if they meet ALL criteria:
- Average phonetic score ≥ 0.6
- All individual phonetic scores > 0.6
- Humor rating ≥ 6 (if available)
- At least 2 tool calls made
- Contains reasoning trace

Customize thresholds:
```bash
python rlvr_dataset_tools.py auto-label \
    --input data.jsonl \
    --output labeled.jsonl \
    --min-phonetic 0.7 \
    --min-humor 7
```
