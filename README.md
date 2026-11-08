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

## Reasoning SFT Pipeline (RunPod)

Trains a Qwen3 LoRA adapter that *reasons* about phonetic swaps inside
`<think>` tags before answering, then generates parodies for the top movies
of 1995-1999. Training data is built from `patruff/chucklesClean720`
(~3,200 human-approved pairs) with suggestions from the
`patruff/parody-suggestions` + `patruff/word-phone` HF tools injected into
every prompt — the same prompt format is used at inference.

### Kick it off from GitHub Actions

Go to **Actions > Reasoning SFT Pipeline > Run workflow**. Leave
**test_mode** checked for the first run (Qwen3-0.6B, ~50 pairs, 20 steps,
a few dollars at most); uncheck it for the full Qwen3-8B run.

The job:
1. Builds the reasoning dataset on the runner (CPU) and pushes it to
   `patruff/chuckles-reasoning-sft`.
2. Rents a RunPod GPU (default RTX 4090, community cloud). The pod clones
   this repo, trains with QLoRA, pushes the adapter to
   `patruff/chuckles-reasoning-adapter`, generates parodies for
   `pipeline/movie_titles_1995_1999.csv`, and uploads its log + results.
3. Polls the Hub until the adapter lands, prints the parodies with their
   tool-verified phonetic scores, **always terminates the pod**, and writes
   elapsed time + estimated GPU cost to the job summary.

Secrets needed: `RUNPOD_API_KEY` and `HUGGINGFACE_API_KEY` (or `HF_TOKEN`)
with **write** scope. `CEREBRAS_API_KEY` is not used by this pipeline.

### Run locally

```bash
export HF_TOKEN=hf_... RUNPOD_API_KEY=...
pip install -e . && pip install runpod

# 1. Build the dataset (CPU)
python pipeline/build_reasoning_dataset.py --limit 50   # drop --limit for all

# 2. Train + infer on a RunPod pod (no SSH needed; pod self-bootstraps)
python pipeline/run_pipeline.py --test    # cheap validation first
python pipeline/run_pipeline.py           # full 8B run

# Emergency teardown if a pod is left running:
python pipeline/run_pipeline.py --down <pod-id>
```

`pipeline/generate_with_reasoning.py` can also run anywhere with a GPU to
try the adapter on any titles CSV.

### Test the Qwen3.8 27B GGUF on RunPod

Select **gguf-test** in the Reasoning SFT Pipeline's **mode** menu, or run the
command below from the repo with `RUNPOD_API_KEY` set. It rents one GPU for
inference with the pinned
[`RVN-Q4_K_M-multilingual.gguf`](https://huggingface.co/0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF)
weights, verifies the download, and terminates the pod after generation.

```bash
pip install -e . 'setuptools<81'
python pipeline/run_gguf_test.py --limit 10 --output-dir out/gguf-01
```

The default sample uses the first ten titles in `pipeline/movie_titles_1995_1999.csv`.
Pass `--titles` to supply another CSV with a `title` column. Use a new output
directory for each run. The default time limit is 25 minutes.

`results.jsonl` includes the parody, model-emitted reasoning, supplied
suggestions, raw assistant message and independently computed phonetic scores
for each title. The scoring tool aligns words by position, so inspect titles
that gain or lose words. Numbers written by the model in its trace can differ
from the scores in `swap_scores`. The run configuration and pod teardown
status are in `run.json`. GitHub Actions collects both files in the
`chuckles-gguf-test` artifact.

### Run the Chuckles smolagents CodeAgent

To replay the saved **The Running Man** demonstration, provide the local
export of `Copy of may18CerebrasQwen3.ipynb` and the March word bank.

```bash
python pipeline/run_gguf_test.py \
  --example-notebook may18CerebrasQwen3.ipynb \
  --march-word-bank ../chuck2025/word_structures.py --cloud SECURE \
  --titles example-titles.csv --limit 1 --output-dir out/gguf-running-man-01
```

Put `The Running Man` under the `title` header of the input CSV. This mode
renders the original creative and verification prompt from cell `tbAjK5rX4tjo`,
using the two matching HF tool revisions, the 0.6 suggestion cutoff, and custom
pronunciations. CodeAgent adds its executable action and result format.
It records every word pair for both finalists, including unchanged words.
The saved cell asks for scores above 0.8 and elsewhere gives 0.6 as the
fallback threshold. Inspect the actual scores for both.

For a humor-first batch, use the other saved cell, `GvGq8GcoGbCD`. This mode
uses the March `parody-suggestions` similarity as an inclusive 0.60 floor
for changed words. It also reports the separate `word-phone` rhyme score.
Unchanged words count as identities and are labeled. Qwen records three or four original
pitches, its own humor ratings, and its two funniest titles before using
`word_phone_tool`. Within the eligible finalists, Qwen picks the winner. If
neither qualifies, it revises; an unsuccessful revision is flagged in the results.

```bash
python pipeline/run_gguf_test.py --humor-first \
  --example-notebook may18CerebrasQwen3.ipynb \
  --march-word-bank ../chuck2025/word_structures.py --cloud SECURE \
  --titles pipeline/humor_sample_titles.csv --limit 10 --timeout-mins 80 --output-dir out/humor-01
python pipeline/export_humor_candidates.py out/humor-01/results.jsonl out/humor-01/candidates.jsonl
```

The export contains one prompt/completion pair per clean agent turn from a
qualifying run, with the model's reasoning and code as separate fields. It
keeps the observations in their original positions in the conversation. All
exported turns start with `review_status: unreviewed`. For a stricter export, install `pip install -e ".[trace-export]"` and pass
`--tokenizer-json TOKENIZER.json --thinking-budget 1536` to omit targets that
come within four tokens of the thinking budget. Supply `--reviews labels.jsonl`
to export only titles marked keep in the matching review file. The raw results keep
failed checks, revisions, and unsuccessful runs for inspection.

The source word bank is [`patruff/chuck2025/word_structures.py`](https://github.com/patruff/chuck2025/blob/783c52912c0be25ccc27fbca768548c46c977a29/word_structures.py)
at SHA-256 `c1ad29501bbc32c0fb8a2a0cefeefe82ca7bb441e1955d87e94fc630ffca9eff`.
It contains 1,565 words and custom pronunciations. The similarly named
`parodies2026/word_structures.py` in this repo has a smaller bank.

To compare the saved Colab code and the complete Running Man suggestion
payload before a GPU run, use the CPU-only check. Pass `--pasted-example` to
compare a saved console excerpt too.

```bash
python pipeline/check_colab_replay.py --example-notebook may18CerebrasQwen3.ipynb \
  --word-bank ../chuck2025/word_structures.py
```

The archived check for `running → cumming` returns 0.665 from the suggestions
tool and 1.000 from word-phone. For `man → klan`, the pair is 0.638 and
0.925. The full word list, rather than the similarity formula, differs in the
January 2026 `parodies2026` copy.

For a paired comparison, `--drafts-from results.jsonl` carries forward each
available title’s original pre-verification pitches and nominations. The model
receives those pitches without the earlier winner or verifier results. It checks
both finalists with the two pinned tools and selects again. Missing titles get
new drafts. Creative and later selection turns retain their original histories.

Use the local export of the `unslothQwQ4bitGOOD` notebook to replay its
“As Good As It Gets” setup on the 27B GGUF. The runner reads the literal
`funny_words` list without executing notebook cells. It uses the 0.8
suggestion cutoff and the two HF tool revisions whose Python source matches
the archived cell output.

```bash
python pipeline/run_gguf_test.py \
  --colab-notebook unslothQwQ4bitGOOD.ipynb --cloud SECURE \
  --titles titles.csv --limit 3 --output-dir out/gguf-colab-01
```

For the March bank of 1,565 words and custom pronunciations, substitute
`--march-word-bank ../chuck2025/word_structures.py`. That profile uses the
existing Hub tools and a 0.6 suggestion cutoff. The March bank is available
in `patruff/chuck2025` at commit `783c529`.

Both profiles use `parody-suggestions` before generation. Qwen writes Python
actions in CodeAgent to call `word_phone_tool`, reads the observations, and
selects its winner with `final_answer`. Each JSONL record contains the input
suggestions, step-by-step code, model reasoning, tool returns, errors,
observations, and the winner and runner-up. Scores are copied from observed
tool returns. Missing pronunciations yield null. A score of 1.12 is possible
in the archived Colab formula. The `passes_threshold` field is a separate
diagnostic for aligned word swaps and never ranks humor.

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
├── pipeline/                  # Reasoning SFT pipeline (RunPod)
│   ├── build_reasoning_dataset.py  # Human pairs -> <think> SFT data -> HF
│   ├── train_reasoning_sft.py      # QLoRA trainer (runs on the pod)
│   ├── generate_with_reasoning.py  # Inference + HF-tool phonetic scoring
│   ├── run_pipeline.py             # Pod lifecycle + cost tracking
│   ├── pod_bootstrap.sh            # Container command run by the pod
│   ├── common.py                   # Shared prompt/trace/scoring helpers
│   └── movie_titles_1995_1999.csv  # Top-10 movies per year, 1995-1999
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
