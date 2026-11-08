# chucklesPRIME

Phonetically-sound parody title generator with DPO preference dataset output.

Generates funny parody titles (e.g. "The Matrix" -> "The Mattress") using AI agents with phonetic verification tools, then provides workflows for human review and DPO (Direct Preference Optimization) dataset creation.

## Qwen 27B on RunPod

The 27B workflow uses the pinned 2025 suggested words and pronunciations. Run
the corrected notebook in Colab or select **gguf-test** in GitHub Actions.
[Open in Colab](https://colab.research.google.com/github/patruff/chuck26/blob/main/notebooks/Chuckles_RunPod.ipynb) · [Notebook source](notebooks/Chuckles_RunPod.ipynb) · [Ten-movie results and reasoning traces](results/2026-11-08-qwen-27b/README.md)

## Teach Qwen what you find funny

Edit [guidance/human-feedback.txt](guidance/human-feedback.txt) and commit it to `main`. That edit starts the [human-guidance Action](https://github.com/patruff/chuck26/actions/workflows/human-guidance.yml) and rents a RunPod GPU for ten fresh titles. You can also open the Action, choose **Run workflow**, and enter the path to another committed `.txt` file under `guidance/`. A single file is all the run needs from you.

Put a prior result, its rationale, or a relevant excerpt of its trace under `MODEL_RUN:`. Add your own comments under `HUMAN_FEEDBACK:`. Both labels start at the beginning of a line. You can add as many example/comment pairs as fit in the model context.

```text
MODEL_RUN:
Top Gun → Pop Gun. The pitch turned the dogfight into a children's party.
HUMAN_FEEDBACK:
The jokes should be more edgy and wild. The recent examples weren't extreme enough.
```

Qwen reads the examples and your comments and adds up to 60 words of creative direction to the original Colab comedy role. Every movie keeps the Colab's humor-first system instruction, explicit audience preferences, and the full ranked list of word suggestions and scores. Short feedback is also quoted verbatim in the generation task. Each movie starts with fresh drafts and uses the verified 2025 suggestions, exact-spelling tool checks, and inclusive 0.60 floor. The revision applies at inference time.

Download the **human-guidance** artifact and open **TRACE-INDEX.md**. Each movie has a readable transcript. It starts with the controller’s actual suggestion lookup, then shows every recorded Qwen thinking block, executed Python, tool return, error, and final choice in order. `prompt-revision.json` shows the original and applied instructions. `results.jsonl` contains each draft, the model's reasoning, the actual system messages and phonetic observations. `candidates.jsonl` contains turns awaiting human review. The artifact also includes the original text and the exact input to the prompt editor. If you paste a very long trace, the editor retains its beginning and end and marks the omission. It preserves all human comments. Comments totaling more than 1,536 tokens inform the revision through the editor alone.

For the next round, copy the artifact's **next-feedback.txt** over `guidance/human-feedback.txt`. It carries forward your comments and the model's previous style instructions, adds the complete step-by-step movie transcripts, and leaves a blank `HUMAN_FEEDBACK:` under each movie. Fill in the movies you want to comment on and commit the file. Newer comments take priority when your preferences change.

The first guided batch used the feedback “more edgy and wild.” Eight of ten movies had qualifying winners. The before-and-after table, recorded prompt revision and editable review file are in the [first guided run](results/2026-11-08-human-guidance/README.md).

A checkout with the RunPod and 2025-bank credentials configured can run the same file locally.

```bash
pip install -e '.[trace-export]' 'smolagents==1.24.0' 'huggingface-hub==0.36.0' 'setuptools<81'
python pipeline/run_gguf_test.py --human-guidance guidance/human-feedback.txt --output-dir out/guided-01
```

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

### Run the Qwen3.8 27B GGUF

**Actions → Reasoning SFT Pipeline → gguf-test** and the CLI use the same
smolagents CodeAgent, corrected Colab prompt and checksum-verified March 2025
bank. The defaults are one Secure Cloud GPU, the ten films in
`pipeline/humor_sample_titles.csv`, and an 80-minute limit. Use
`infer_limit=1` for a smoke test. The workflow uploads its run manifest,
generation traces and screened training candidates as the `chuckles-gguf-test`
artifact. **gguf-preflight** runs the historical controls on CPU.

```bash
pip install -e '.[trace-export]' 'smolagents==1.24.0' 'huggingface-hub==0.36.0' 'setuptools<81'
python pipeline/run_gguf_test.py --preflight-only
# Set RUNPOD_API_KEY in the environment before the GPU run.
python pipeline/run_gguf_test.py --limit 1 --output-dir out/chuckles-01
python pipeline/export_humor_candidates.py out/chuckles-01/results.jsonl out/chuckles-01/candidates.jsonl \
  --run-manifest out/chuckles-01/run.json
```

The bank comes from `word_structures.py` at commit `783c529` in
`patruff/chuck2025`. The launcher accepts `--march-word-bank PATH`, an
encrypted Actions secret named `CHUCKLES_2025_BANK_GZ` (base64 of the gzip
file), a sibling `chuck2025` checkout, `CHUCK2025_READ_TOKEN`, or an
authenticated GitHub CLI. It verifies SHA-256
`c1ad29501bbc32c0fb8a2a0cefeefe82ca7bb441e1955d87e94fc630ffca9eff`
and checks the original Running Man suggestions and scorer values before
renting a GPU. The private source stays on the controller. The smaller
`parodies2026/word_structures.py` in this repo has 226 words and four custom
pronunciations. [Canonical bank](https://github.com/patruff/chuck2025/blob/783c52912c0be25ccc27fbca768548c46c977a29/word_structures.py)

To provision the Actions secret in another checkout, use an authenticated
GitHub CLI with permission to set repository secrets:

```bash
gzip -c ../chuck2025/word_structures.py | base64 | tr -d '\n' | gh secret set CHUCKLES_2025_BANK_GZ --repo patruff/chuck26
```

Qwen records three or four distinct pitches, each with a humor rating and
comic premise, then nominates its funniest pitch using a returned swap and its funniest remaining word-aligned pitch. CodeAgent
checks the exact spelling of each word in both finalists. Each changed word
needs at least 0.60 from the pinned March `parody-suggestions` formula.
`word-phone` supplies an independent rhyme/length/stress score. Once both
titles qualify, a title using a listed change takes priority over an entirely off-list title. Qwen chooses on humor between equally eligible titles. With no returned changes it can invent its own. To skip available suggestions it must name a returned word and explain why the suggestions fall flat. The transcript records that exception. If revised finalists also fail,
`qualifying_parody` is null and `fallback` identifies the model's best failed
attempt. Only observed, qualifying results can enter the review-candidate export.

The archived tool check gives 0.665 and 1.000 for *running → cumming*.
*Running → huning* gets 0.830 and 1.000; *running → hunning* is unscorable
in both tools. Each draft must explain what its changed phrase means and what
makes the scene funny. The full controls and the rejected *Hunning* example
are in the [batch findings](results/2026-11-08-qwen-27b/README.md).

To replay the original Drive notebook, pass its local export with
`--example-notebook`; add `--humor-first` for the two-phase batch.
`--colab-notebook` selects the separate historical 0.80/321-entry experiment,
and `--base-only` selects the old single-answer baseline. For a paired
comparison, `--drafts-from results.jsonl` reuses only matching, pre-verification
pitches. To compare the pinned tools' Python and complete Running Man input
with the saved original notebook, run:

```bash
python pipeline/check_colab_replay.py --example-notebook may18CerebrasQwen3.ipynb \
  --word-bank ../chuck2025/word_structures.py
```

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
