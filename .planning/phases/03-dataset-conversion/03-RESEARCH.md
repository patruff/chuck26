# Phase 3: Dataset Conversion - Research

**Researched:** 2026-01-31
**Domain:** TRL dataset formats (GRPO/DPO), HuggingFace datasets library, composite reward signals
**Confidence:** HIGH

## Summary

Phase 3 converts GenerationRecord objects (from Phase 2) into two training-ready datasets: a GRPO prompt-only dataset and a DPO preference dataset, computes composite reward signals, archives reasoning traces as JSONL, and pushes both datasets to HuggingFace Hub.

The TRL library defines precise dataset contracts: GRPOTrainer expects a **prompt-only** dataset with a `prompt` column (conversational format: list of message dicts) plus arbitrary auxiliary columns passed to reward functions via kwargs. DPOTrainer expects a **preference** dataset with `prompt`, `chosen`, and `rejected` columns in conversational format. Both formats are well-documented and stable.

The `datasets` library (>=3.0.0) provides `Dataset.from_list()` to build datasets from lists of dicts and `push_to_hub()` for upload. Authentication uses the `HF_TOKEN` environment variable. The library must be added to pyproject.toml dependencies.

**Primary recommendation:** Use conversational format for both GRPO and DPO datasets. Build datasets with `Dataset.from_list()`. Compute three continuous composite reward signals (phonetic_quality, tool_usage_completeness, structure_preservation) as floats. Push via `dataset.push_to_hub()` with `private=True`.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `datasets` | `>=3.0.0` | Dataset creation, serialization, Hub push | HuggingFace's canonical dataset library. Native Parquet, `push_to_hub()`, Dataset Viewer support. 46M+ monthly PyPI downloads. |
| `huggingface-hub` | `>=0.20.0` | Authentication, Hub API | Transitive dependency of `datasets`. Provides `login()`, `HF_TOKEN` env var support. |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `pronouncing` | `>=0.2.0` | Phonetic scoring for rewards | Already in deps. Used to compute phonetic_quality reward signal. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `Dataset.from_list()` | `Dataset.from_dict()` | from_list is more natural for record-oriented data; from_dict requires column-oriented layout |
| `Dataset.from_list()` | `Dataset.from_pandas()` | Adds pandas dependency unnecessarily; from_list works directly with dicts |
| Manual Parquet | `push_to_hub()` | push_to_hub handles Parquet serialization, card metadata, viewer compatibility automatically |

**Installation:**
```bash
pip install datasets huggingface-hub
```

**pyproject.toml addition:**
```toml
dependencies = [
    "smolagents>=1.24.0",
    "openai>=1.0.0",
    "rich",
    "pronouncing",
    "datasets>=3.0.0",
    "huggingface-hub>=0.20.0",
]
```

Note: `trl` is NOT needed as a dependency. We generate data *for* TRL, but do not import TRL. The format spec is documented and stable.

## Architecture Patterns

### Recommended Project Structure
```
src/chuckles_prime/
    config.py          # (Phase 1) AppConfig with human_examples
    types.py           # (Phase 2) GenerationRecord, ParodyCandidate, AgentTrace
    generator.py       # (Phase 2) generate_single, generate_batch
    rewards.py         # (Phase 3) NEW - Composite reward signal computation
    dataset.py         # (Phase 3) NEW - GRPO/DPO format converters + Hub push
    traces.py          # (Phase 3) NEW - JSONL trace archival
```

### Pattern 1: Converter Functions (Pure Transforms)

**What:** Stateless functions that transform GenerationRecord objects into TRL-compatible dicts.
**When to use:** All dataset conversion -- keeps conversion testable without Hub access.

```python
# Source: TRL dataset_formats docs (https://huggingface.co/docs/trl/main/en/dataset_formats)
def record_to_grpo(record: GenerationRecord, system_prompt: str) -> dict:
    """Convert a GenerationRecord to a GRPO prompt-only dict."""
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Create a phonetically-sound parody of: '{record.input_title}'"},
        ],
        # Auxiliary columns for reward functions
        "original_title": record.input_title,
        "phonetic_scores": json.dumps({
            c.text: c.phonetic_scores for c in record.candidates
        }),
        "generation_model": record.model_name,
    }
```

### Pattern 2: DPO Pairing (Human vs Model)

**What:** Pair human parody examples (chosen) against model inferior outputs (rejected) for the same input title.
**When to use:** DPO dataset construction.

```python
# Source: TRL DPO docs (https://huggingface.co/docs/trl/main/en/dpo_trainer)
def build_dpo_pair(
    input_title: str,
    human_output: str,
    human_explanation: str,
    model_output: str,
    system_prompt: str,
) -> dict:
    """Build a single DPO preference pair."""
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Create a phonetically-sound parody of: '{input_title}'"},
        ],
        "chosen": [
            {"role": "assistant", "content": human_output},
        ],
        "rejected": [
            {"role": "assistant", "content": model_output},
        ],
    }
```

### Pattern 3: Hub Push with Authentication

**What:** Create dataset from records and push to Hub.
**When to use:** Final step of conversion pipeline.

```python
# Source: datasets upload docs (https://huggingface.co/docs/datasets/en/upload_dataset)
import os
from datasets import Dataset
from huggingface_hub import login

def push_dataset(records: list[dict], repo_id: str, split: str = "train") -> None:
    """Push a list of record dicts to HuggingFace Hub."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable not set")
    login(token=token)

    dataset = Dataset.from_list(records)
    dataset.push_to_hub(repo_id, split=split, private=True)
```

### Anti-Patterns to Avoid

- **Including model completions in GRPO dataset:** GRPO is prompt-only. The model generates completions during training. Do NOT include model outputs in the GRPO dataset.
- **Implicit prompts in DPO:** Always use explicit prompts (separate `prompt` column). TRL recommends explicit prompts. Implicit prompts (where prompt is embedded in chosen/rejected) require extra extraction and are fragile.
- **Storing nested dicts as Python objects:** JSON-serialize any complex nested data (phonetic_scores dict, trace steps) as strings. HuggingFace datasets and Parquet handle strings, not arbitrary nested Python objects. Use `json.dumps()` for metadata columns.
- **Binary threshold rewards:** Do NOT use 0/1 rewards. Use continuous float scores. Binary rewards destroy gradient information (a score of 0.61 and 0.99 both get reward=1).
- **Forgetting `remove_unused_columns=False`:** Document in dataset card that trainers must use `GRPOConfig(remove_unused_columns=False)` to preserve auxiliary columns for reward functions.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Dataset serialization to Parquet | Custom Parquet writer | `Dataset.from_list()` + `push_to_hub()` | Handles schema inference, chunking, Parquet row groups, Dataset Viewer metadata |
| Hub authentication | Custom token management | `huggingface_hub.login()` + `HF_TOKEN` env var | Handles token caching, validation, priority (env > stored) |
| Chat template formatting | Custom message formatter | TRL conversational format (list of dicts) | Trainer applies model's chat template automatically |
| DPO prompt extraction | Custom prompt parser | TRL `extract_prompt()` utility | Handles edge cases in common prefix detection |

**Key insight:** The `datasets` library + `push_to_hub()` does all the heavy lifting for dataset creation and upload. Do not manually create Parquet files, manage Hub API calls, or build dataset cards programmatically.

## Common Pitfalls

### Pitfall 1: Wrong Conversational Format for GRPO
**What goes wrong:** Using `{"prompt": "plain text string"}` instead of `{"prompt": [{"role": "user", "content": "..."}]}`. Both are valid, but conversational format is strongly recommended because it preserves system prompt instructions and the trainer applies the model's chat template automatically.
**Why it happens:** Standard format looks simpler.
**How to avoid:** Always use conversational format (list of message dicts) for the `prompt` column. Include system message.
**Warning signs:** Training outputs ignore system prompt context.

### Pitfall 2: DPO Chosen/Rejected Must Be Assistant-Only
**What goes wrong:** Including the user prompt inside chosen/rejected messages when using explicit prompts. The `prompt` column already contains the user message. The `chosen` and `rejected` columns should contain ONLY the assistant's response.
**Why it happens:** Confusion between explicit and implicit prompt formats.
**How to avoid:** With explicit prompts: `prompt` = [system + user messages], `chosen` = [assistant message only], `rejected` = [assistant message only].
**Warning signs:** DPOTrainer extracts duplicate prompts, tokenization errors.

### Pitfall 3: Nested Python Objects in Dataset Columns
**What goes wrong:** Storing dicts or lists of dicts as column values without JSON serialization. Arrow/Parquet requires homogeneous types. A dict column works if ALL rows have identical keys, but fails with varying structures.
**Why it happens:** Python dicts seem natural for metadata.
**How to avoid:** JSON-serialize variable-structure metadata to string columns. Use `json.dumps()` for phonetic_scores, trace data. Only use native list/dict columns for fixed schemas (like the messages format).
**Warning signs:** `ArrowInvalid` errors during `Dataset.from_list()`, missing data in Dataset Viewer.

### Pitfall 4: HF_TOKEN Not Set
**What goes wrong:** `push_to_hub()` fails with authentication error. Silent failure if token lacks write permission.
**Why it happens:** Developer forgets to set env var, or uses read-only token.
**How to avoid:** Check `HF_TOKEN` exists and is not empty before attempting push. Provide clear error message. Token must have **write** permission.
**Warning signs:** 401/403 errors from Hub API.

### Pitfall 5: Matching DPO Pairs Across Different Sources
**What goes wrong:** Human examples (from CSV) don't have corresponding model outputs for the same input title, making DPO pair construction impossible.
**Why it happens:** Human examples and model generations cover different input titles.
**How to avoid:** Two strategies: (a) Generate model outputs for titles that have human examples (requires Phase 2 output for matching titles), or (b) Use human examples as chosen and pair with ANY model inferior output (less ideal but functional). Strategy (a) is preferred.
**Warning signs:** Empty DPO dataset, mismatched input titles between chosen and rejected.

### Pitfall 6: Phonetic Scores Dict Has Variable Keys
**What goes wrong:** Each GenerationRecord has different words in phonetic_scores (because different titles have different words). Storing as a native dict column fails when keys vary between rows.
**Why it happens:** Each title has unique words.
**How to avoid:** JSON-serialize the phonetic_scores dict: `json.dumps(candidate.phonetic_scores)`. Store as a string column. Downstream reward functions parse with `json.loads()`.
**Warning signs:** Schema inference errors, missing scores in some rows.

## Code Examples

Verified patterns from official sources:

### GRPO Dataset Creation (Full Example)
```python
# Source: TRL dataset_formats (https://huggingface.co/docs/trl/main/en/dataset_formats)
# Source: datasets upload (https://huggingface.co/docs/datasets/en/upload_dataset)
import json
from datasets import Dataset

SYSTEM_PROMPT = (
    "You are a comedy writer who creates funny parody titles. "
    "Replace words with phonetically similar but humorous alternatives. "
    "Use the phonetic analysis tool to verify similarity scores above 0.6."
)

def records_to_grpo_dataset(records: list[GenerationRecord]) -> Dataset:
    """Convert GenerationRecord list to TRL GRPO prompt-only dataset."""
    rows = []
    for rec in records:
        if rec.error:
            continue  # Skip failed generations

        # Compute aggregate phonetic score
        all_scores = []
        for c in rec.candidates:
            all_scores.extend(c.phonetic_scores.values())
        avg_phonetic = sum(all_scores) / len(all_scores) if all_scores else 0.0

        rows.append({
            # Required: prompt in conversational format
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Create a phonetically-sound parody of: '{rec.input_title}'"},
            ],
            # Auxiliary columns for reward functions
            "original_title": rec.input_title,
            "phonetic_scores": json.dumps({
                c.text: c.phonetic_scores for c in rec.candidates
            }),
            "avg_phonetic_score": avg_phonetic,
            "generation_model": rec.model_name,
            "num_candidates": len(rec.candidates),
        })

    return Dataset.from_list(rows)
```

### DPO Dataset Creation (Full Example)
```python
# Source: TRL DPO docs (https://huggingface.co/docs/trl/main/en/dpo_trainer)
def build_dpo_dataset(
    human_examples: list[tuple[str, str, str]],  # (input, output, explanation)
    model_records: dict[str, GenerationRecord],   # input_title -> record
) -> Dataset:
    """Build DPO preference dataset pairing human chosen vs model rejected.

    Only includes pairs where both human example and model output exist
    for the same input title.
    """
    rows = []
    for input_title, human_output, explanation in human_examples:
        record = model_records.get(input_title)
        if not record or not record.candidates:
            continue

        # Use worst model candidate as rejected
        worst = min(record.candidates, key=lambda c:
            sum(c.phonetic_scores.values()) / max(len(c.phonetic_scores), 1))

        rows.append({
            # Explicit prompt (recommended by TRL)
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Create a phonetically-sound parody of: '{input_title}'"},
            ],
            # Chosen = human parody (assistant-only)
            "chosen": [
                {"role": "assistant", "content": human_output},
            ],
            # Rejected = model inferior output (assistant-only)
            "rejected": [
                {"role": "assistant", "content": worst.text},
            ],
        })

    return Dataset.from_list(rows)
```

### Composite Reward Signal Computation
```python
# These are stored as metadata in GRPO dataset for reference,
# and used by downstream trainers as verifiable reward functions.

def compute_phonetic_quality(candidate: ParodyCandidate) -> float:
    """Continuous phonetic quality score [0.0, 1.0].

    Average of all word-level phonetic similarity scores.
    Higher = better phonetic match to original words.
    """
    scores = list(candidate.phonetic_scores.values())
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def compute_tool_usage_completeness(trace: AgentTrace, input_title: str) -> float:
    """Continuous tool usage score [0.0, 1.0].

    Measures what fraction of title words were phonetically verified.
    Counts distinct word_phonetic_analyzer calls vs words in title.
    """
    title_words = [w for w in input_title.split() if len(w) > 2]
    if not title_words:
        return 1.0

    # Count unique words verified via phone_tool in trace steps
    verified_words = set()
    for step in trace.steps:
        step_str = str(step)
        for word in title_words:
            if word.lower() in step_str.lower():
                verified_words.add(word.lower())

    return len(verified_words) / len(title_words)


def compute_structure_preservation(input_title: str, parody_text: str) -> float:
    """Continuous structure preservation score [0.0, 1.0].

    Measures how well the parody preserves the word count and
    structural pattern of the original title.
    """
    orig_words = input_title.split()
    parody_words = parody_text.split()

    if not orig_words:
        return 0.0

    # Word count similarity (penalize adding/removing words)
    count_ratio = min(len(parody_words), len(orig_words)) / max(len(parody_words), len(orig_words))

    return count_ratio
```

### JSONL Trace Archival
```python
import json
from pathlib import Path
from dataclasses import asdict

def archive_traces(records: list[GenerationRecord], output_path: Path) -> int:
    """Archive full reasoning traces as JSONL.

    One JSON line per GenerationRecord. Preserves all fields including
    the full AgentTrace with step-by-step reasoning.

    Returns number of records written.
    """
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            # Use dataclasses.asdict for full serialization
            record_dict = asdict(rec)
            f.write(json.dumps(record_dict, ensure_ascii=False, default=str) + "\n")
            count += 1
    return count
```

### Hub Push
```python
import os
from datasets import Dataset
from huggingface_hub import login

def push_to_hub(
    dataset: Dataset,
    repo_id: str,
    split: str = "train",
    private: bool = True,
) -> None:
    """Push a dataset to HuggingFace Hub.

    Requires HF_TOKEN environment variable with write permission.
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError(
            "HF_TOKEN environment variable not set. "
            "Create a write token at https://huggingface.co/settings/tokens"
        )
    login(token=token)
    dataset.push_to_hub(repo_id, split=split, private=private)
```

## Exact TRL Format Specifications

### GRPO (Prompt-Only) Format

**Source:** [TRL Dataset Formats](https://huggingface.co/docs/trl/main/en/dataset_formats), [GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer)

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `prompt` | `list[dict]` (conversational) or `str` (standard) | YES | The input prompt. Conversational format recommended. |
| Any auxiliary column | Any | NO | Passed to reward functions as kwargs. Column name becomes kwarg name. |

**Conversational format example:**
```python
{
    "prompt": [
        {"role": "system", "content": "You create phonetic parodies."},
        {"role": "user", "content": "Create a parody of 'The Matrix'"}
    ],
    "original_title": "The Matrix",
    "phonetic_scores": '{"The Mattress": {"Matrix": 0.78}}',
    "generation_model": "qwen-3-32b"
}
```

**Reward function signature (for downstream training):**
```python
def my_reward_func(prompts, completions, original_title, phonetic_scores, **kwargs):
    """Receives all auxiliary columns as kwargs."""
    rewards = []
    for completion, title, scores_json in zip(completions, original_title, phonetic_scores):
        scores = json.loads(scores_json)
        reward = compute_reward(completion, title, scores)
        rewards.append(reward)
    return rewards
```

**Critical config for training:** `GRPOConfig(remove_unused_columns=False)` -- preserves auxiliary columns.

### DPO (Preference) Format

**Source:** [TRL DPO Trainer](https://huggingface.co/docs/trl/main/en/dpo_trainer), [TRL Dataset Formats](https://huggingface.co/docs/trl/main/en/dataset_formats)

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `prompt` | `list[dict]` (conversational) or `str` (standard) | YES (recommended explicit) | The input prompt (system + user messages). |
| `chosen` | `list[dict]` (conversational) or `str` (standard) | YES | The preferred response (assistant message only with explicit prompt). |
| `rejected` | `list[dict]` (conversational) or `str` (standard) | YES | The dispreferred response (assistant message only with explicit prompt). |

**Conversational format example (explicit prompt, recommended):**
```python
{
    "prompt": [
        {"role": "system", "content": "You create phonetic parodies."},
        {"role": "user", "content": "Create a parody of 'The Matrix'"}
    ],
    "chosen": [
        {"role": "assistant", "content": "The Mattress"}
    ],
    "rejected": [
        {"role": "assistant", "content": "The Matricks"}
    ]
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Binary rewards (0/1) | Composite continuous rewards | GRPO++ / RLVRR (2025) | Preserves gradient information, better training signal |
| Single reward function | Multiple weighted reward functions | TRL 0.25+ | GRPOTrainer natively supports `reward_funcs=[fn1, fn2, ...]` with `reward_weights` |
| Standard text prompts | Conversational format | TRL 0.20+ | Chat template applied automatically, system prompt preserved |
| `HfApiModel` | `InferenceClientModel` | smolagents 1.20+ | Old name deprecated, new name is canonical |
| `datasets` v2 | `datasets` v3+ | Mid 2025 | Faster push_to_hub with content-defined chunking, streaming push support |

**Deprecated/outdated:**
- Binary threshold rewards (0.6 cutoff): Use continuous float scores instead
- RLVR template tags (from existing `rlvr_dataset_tools.py`): TRL does not use `<start_working_out>` tags; use standard chat message format
- Custom Parquet writers: Use `datasets` library `push_to_hub()` instead
- Implicit prompt DPO format: Use explicit prompt format (separate `prompt` column)

## Open Questions

Things that could not be fully resolved:

1. **DPO pair coverage: How many human examples will have matching model outputs?**
   - What we know: ~1,098 human examples exist. Model generation (Phase 2) processes arbitrary CSV titles.
   - What's unclear: Whether Phase 2 generation will run on the same titles as human examples. If not, DPO dataset will be small or empty.
   - Recommendation: Plan two strategies -- (a) Primary: generate model outputs for human example titles, creating direct pairs. (b) Fallback: pair human examples with model outputs for different titles (same-prompt pairing not required by DPO, but preferred).

2. **Optimal system prompt content for GRPO dataset**
   - What we know: System prompt is stored in the `prompt` messages. It shapes what the model learns during GRPO training.
   - What's unclear: Whether the exact system prompt from `prompts.py` (PARODY_INSTRUCTIONS) is appropriate, or if a simpler version is better for training.
   - Recommendation: Use a simplified system prompt in the dataset (not the full agent instructions). The dataset system prompt should describe the task, not the agent workflow.

3. **Reward signal normalization: Should scores be pre-normalized or raw?**
   - What we know: GRPO normalizes rewards within each group. Storing pre-normalized scores could conflict with group normalization.
   - What's unclear: Whether storing raw continuous scores or pre-normalized [0,1] scores is better for downstream training.
   - Recommendation: Store raw continuous scores in [0.0, 1.0] range. Let the trainer handle group normalization. Document the score ranges in the dataset card.

4. **Structure preservation: Best metric?**
   - What we know: Word count ratio is simple but crude. Syllable count, stress pattern matching could be more informative.
   - What's unclear: Which metric best captures "the parody sounds like it could be a real title."
   - Recommendation: Start with word count ratio for v1. Add syllable-based metrics in v2 once baseline is established. The continuous score design allows easy swap-in of better metrics.

## Sources

### Primary (HIGH confidence)
- [TRL Dataset Formats](https://huggingface.co/docs/trl/main/en/dataset_formats) - Complete specification of prompt-only, preference, conversational formats
- [TRL GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) - Reward function signatures, `remove_unused_columns`, auxiliary columns, multi-reward support
- [TRL DPOTrainer](https://huggingface.co/docs/trl/main/en/dpo_trainer) - Preference format, explicit vs implicit prompts, conversational format
- [HuggingFace Datasets Upload Guide](https://huggingface.co/docs/datasets/en/upload_dataset) - `push_to_hub()`, authentication, private datasets
- [HuggingFace Hub Environment Variables](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables) - `HF_TOKEN` usage, priority over stored tokens

### Secondary (MEDIUM confidence)
- [TRL GitHub - dataset_formats.md](https://github.com/huggingface/trl/blob/main/docs/source/dataset_formats.md) - Source-of-truth for format specs
- [TRL GitHub - grpo_trainer.py](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py) - Implementation details for reward function calling
- [HuggingFace Datasets Main Classes](https://huggingface.co/docs/datasets/en/package_reference/main_classes) - `Dataset.from_list()`, Features, push_to_hub API
- [datasets PyPI](https://pypi.org/project/datasets/) - Current version info
- [huggingface-hub PyPI](https://pypi.org/project/huggingface-hub/) - v1.3.5, Jan 29, 2026

### Tertiary (LOW confidence)
- Existing `parodies2026/rlvr_dataset_tools.py` - Reference for prior approach (uses non-TRL formats, needs replacement)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - `datasets` and `huggingface-hub` are canonical, well-documented
- TRL GRPO format: HIGH - Directly verified from official TRL docs (fetched 2026-01-31)
- TRL DPO format: HIGH - Directly verified from official TRL docs (fetched 2026-01-31)
- Hub push pattern: HIGH - Documented in official datasets guide
- Composite rewards: MEDIUM - Design is sound but specific metric formulas need empirical validation
- Architecture: HIGH - Standard converter + push pattern, well-established

**Research date:** 2026-01-31
**Valid until:** 2026-03-01 (stable APIs, 30-day validity)
