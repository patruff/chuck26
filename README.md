# chucklesPRIME

Phonetically-sound parody title generator with DPO preference dataset output.

Generates funny parody titles (e.g. "The Matrix" -> "The Mattress") using AI agents with phonetic verification tools, then provides workflows for human review and DPO (Direct Preference Optimization) dataset creation.

## Generating a DPO Dataset: Step-by-Step

There are two workflows for reviewing parodies and building DPO data: **GitHub PR-based** and **Google Drive-based**. Both track provenance (model name, adapter/LoRA) for every candidate.

---

### Option A: GitHub PR Review Workflow

Best for async team review via pull requests.

#### Prerequisites

Add these secrets to your GitHub repository (Settings > Secrets > Actions):

| Secret | Purpose |
|--------|---------|
| `CEREBRAS_API_KEY` | API key for the generation model |
| `HF_TOKEN` | HuggingFace token with write access |

#### Step 1: Trigger Generation

Go to **Actions > Generate Parodies for Review > Run workflow** and configure:

- **Model**: e.g. `qwen-3-32b` (default)
- **Adapter**: LoRA adapter name, if any (leave empty for base model)
- **Input CSV**: path to CSV with titles (default: `titles.csv`)
- **Titles count**: number of titles to sample (default: 10)

This generates parodies and opens a PR with a review CSV.

#### Step 2: Review the CSV

Open the PR. The CSV in `reviews/pending/` has these columns:

| Column | Description |
|--------|-------------|
| `id` | Row identifier |
| `input_title` | Original title |
| `parody_text` | Generated parody |
| `humor_note` | Model's humor explanation |
| `phonetic_scores` | Per-word phonetic similarity (JSON) |
| `avg_phonetic_score` | Average phonetic score |
| `model_name` | Which model generated this |
| `adapter` | Which adapter/LoRA was used |
| `status` | **Edit this**: `chosen`, `rejected`, or leave as `pending` |

Edit the `status` column for each row:

- **`chosen`** - Good parody (will be the preferred response in DPO)
- **`rejected`** - Bad parody (will be the dispreferred response in DPO)
- **`pending`** - Skip this row

For each `input_title`, you need at least one `chosen` and one `rejected` to form a DPO pair. Every (chosen, rejected) combination for the same title becomes one training example.

#### Step 3: Merge the PR

Commit your edits and merge. The **Process Reviews** workflow automatically:

1. Reads the CSV and groups rows by `input_title`
2. Forms DPO pairs from all (chosen, rejected) combinations per title
3. Appends pairs to the HuggingFace DPO dataset (default: `patruff/chuckles-dpo`)
4. Moves the CSV to `reviews/processed/`

#### Step 4: Verify

Check your HuggingFace dataset. Each DPO row includes:

```json
{
  "prompt": [
    {"role": "system", "content": "You are a comedy writer..."},
    {"role": "user", "content": "Create a phonetically-sound parody of: 'The Matrix'"}
  ],
  "chosen": [{"role": "assistant", "content": "The Mattress"}],
  "rejected": [{"role": "assistant", "content": "The Madness"}],
  "chosen_model": "qwen-3-32b",
  "chosen_adapter": "",
  "rejected_model": "qwen-3-32b",
  "rejected_adapter": "my-lora-v1",
  "chosen_phonetic_score": "0.820",
  "rejected_phonetic_score": "0.710"
}
```

---

### Option B: Google Drive Review Workflow

Best for reviewing directly in Google Sheets.

#### Prerequisites

1. Create a Google Cloud service account with Drive API access
2. Share a `chuck26` folder with the service account email
3. Add these secrets to GitHub (or set as environment variables locally):

| Secret / Env Var | Purpose |
|------------------|---------|
| `GOOGLE_DRIVE_CREDENTIALS` | Full JSON of the service account key |
| `CEREBRAS_API_KEY` | API key for the generation model |
| `HF_TOKEN` | HuggingFace token with write access |

#### Step 1: Add Input Titles

Upload a CSV with a `title` column to `chuck26/input/` in Google Drive:

```csv
title
The Matrix
Die Hard
Fight Club
Top Gun
```

#### Step 2: Generate Parodies

**Via GitHub Actions:**

Go to **Actions > Drive Review App > Run workflow**, select `generate`, and configure the model/adapter/limit.

**Or run locally:**

```bash
pip install -e ".[drive]"
export GOOGLE_DRIVE_CREDENTIALS='{"type":"service_account",...}'
export CEREBRAS_API_KEY="your-key"

python scripts/drive_review_app.py generate --model qwen-3-32b --limit 10
```

Generated review CSVs appear in `chuck26/to_be_checked/`.

#### Step 3: Review in Google Sheets

Open the CSV in `chuck26/to_be_checked/` with Google Sheets. Edit the `status` column:

- `chosen` for good parodies
- `rejected` for bad parodies
- Leave as `pending` to skip

Save the file (keep it as CSV).

#### Step 4: Process Reviews

**Via GitHub Actions:**

Go to **Actions > Drive Review App > Run workflow**, select `process`.

**Or run locally:**

```bash
export HF_TOKEN="your-hf-token"
python scripts/drive_review_app.py process --dpo-repo patruff/chuckles-dpo
```

This:
1. Downloads reviewed CSVs from `to_be_checked/`
2. Builds DPO preference pairs
3. Appends to the HuggingFace dataset
4. Moves processed CSVs to `chuck26/finished_preference/`

#### Step 5: Check Status

See what's in each folder:

```bash
python scripts/drive_review_app.py status
```

Output:
```
chuck26/input/ (1 CSV files)
  titles.csv

chuck26/to_be_checked/ (0 CSV files)
  (empty)

chuck26/finished_preference/ (1 CSV files)
  review-titles-20260131-143022-done.csv
```

---

## Running Locally (No Google Drive)

You can also run generation + review entirely on your local filesystem:

```bash
# Generate review CSV
python scripts/generate_for_review.py titles.csv \
    --output-dir reviews/pending \
    --model qwen-3-32b \
    --adapter my-lora

# Edit reviews/pending/review-*.csv manually (change status column)

# Process into DPO data
python scripts/process_reviews.py \
    --reviews-dir reviews/pending \
    --dpo-repo patruff/chuckles-dpo
```

---

## Project Structure

```
chuck26/
├── src/chuckles_prime/        # Core package
│   ├── cli.py                 # CLI: generate, convert, label, export-labels
│   ├── generator.py           # Parody generation engine
│   ├── dataset.py             # GRPO/DPO dataset builders
│   ├── labeler.py             # Flask web UI for labeling
│   ├── model.py               # OpenAI-compatible model adapter
│   ├── config.py              # Settings loader
│   ├── prompts.py             # Prompt templates
│   ├── rewards.py             # Phonetic quality reward functions
│   ├── tools.py               # HuggingFace tool loader
│   ├── traces.py              # JSONL trace archival
│   └── types.py               # ParodyCandidate, AgentTrace, GenerationRecord
│
├── scripts/
│   ├── generate_for_review.py # Generate parodies -> review CSV
│   ├── process_reviews.py     # Review CSV -> DPO dataset -> HuggingFace
│   └── drive_review_app.py    # Google Drive three-folder workflow
│
├── .github/workflows/
│   ├── generate-review.yml    # Generate parodies + open review PR
│   ├── process-reviews.yml    # Process merged reviews -> HuggingFace DPO
│   └── drive-review.yml       # Google Drive workflow (generate/process/status)
│
├── reviews/
│   ├── pending/               # Review CSVs awaiting human review
│   └── processed/             # Archived after DPO export
│
├── tests/                     # pytest test suite
├── titles.csv                 # Default input titles
├── parodies2026/              # Legacy prototype
└── pyproject.toml
```

## Review CSV Format

All workflows use the same CSV format:

```csv
id,input_title,parody_text,humor_note,phonetic_scores,avg_phonetic_score,model_name,adapter,status
1,The Matrix,The Mattress,Sci-fi becomes furniture,{"Matrix": 0.82},0.820,qwen-3-32b,,pending
2,The Matrix,The Madness,Mental health parody,{"Matrix": 0.71},0.710,qwen-3-32b,,pending
```

## DPO Pair Formation

For each `input_title`, every (chosen, rejected) combination produces one DPO training pair:

| chosen | rejected | pairs |
|--------|----------|-------|
| 1 chosen, 1 rejected | 1 pair |
| 2 chosen, 1 rejected | 2 pairs |
| 1 chosen, 2 rejected | 2 pairs |
| 2 chosen, 2 rejected | 4 pairs |

Provenance metadata is preserved so you can analyze which models/adapters produce preferred outputs.

## Setup

```bash
# Install core package
pip install -e .

# Install with Google Drive support
pip install -e ".[drive]"

# Install dev dependencies (tests, linting)
pip install -e ".[dev]"

# Run tests
pytest tests/
```

## Required Environment Variables

| Variable | Used By | Purpose |
|----------|---------|---------|
| `CEREBRAS_API_KEY` | Generation | API key for Cerebras inference |
| `HF_TOKEN` | DPO push | HuggingFace write token |
| `GOOGLE_DRIVE_CREDENTIALS` | Drive workflow | Service account JSON |
