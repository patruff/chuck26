# Architecture Research

**Research Date:** 2026-01-31
**Scope:** chucklesPRIME restructuring -- config injection, module layout, LLM adapter, RLVR data flow, human examples, build order.

---

## System Components

### Component Boundaries

The system has six distinct components with clean boundaries:

**1. Config Layer** (external, opaque)
- `funny_words.json` -- word lists the model uses for parody candidates
- `preferences.json` -- user humor style description, injected into prompts verbatim
- `human_examples.csv` -- thousands of input/output parody pairs
- These files live outside the repo. The app loads them at startup and passes their contents through without parsing or interpreting structure.

**2. LLM Adapter Layer** (swappable backend)
- Wraps any chat-completion-compatible LLM for use with smolagents
- Current: custom `CerebrasModel` class with `__call__` interface
- Target: use smolagents' built-in `LiteLLMModel` or `OpenAIModel` -- no custom adapter needed
- The adapter is the only component that touches the LLM API

**3. Agent Orchestration Layer** (smolagents CodeAgent)
- Receives a fully-constructed prompt (from Prompt Builder)
- Has access to `word_phone_tool` for phonetic verification
- Runs multi-step reasoning: brainstorm -> verify -> select
- Returns raw text output with reasoning traces and tool calls embedded

**4. Prompt Builder** (assembles context for the agent)
- Takes: title, config data (funny words, preferences), human examples, pre-computed suggestions
- Produces: a single prompt string that the agent executes
- This is where human examples and style preferences get injected into the LLM context

**5. Output Parser + RLVR Converter** (post-processing)
- Extracts structured data from raw agent output (thinking trace, tool calls, final parody, attempts)
- Computes quality signals (phonetic scores, tool usage count)
- Converts to TRL-compatible dataset format (prompt-only for GRPO, preference for DPO)
- Handles both single-item and batch conversion

**6. Pipeline Orchestrator** (entry point + glue)
- Reads input CSV, loads configs, initializes LLM adapter
- For each title: runs suggestion pre-computation, builds prompt, executes agent, parses output
- Collects all results into RLVR dataset
- Pushes to HuggingFace Hub

### What Changes vs. Current

| Current (`parodies2026/`) | New (`chucklesPRIME`) |
|---|---|
| `CerebrasModel` custom adapter class | Use `LiteLLMModel` or `InferenceClientModel` from smolagents |
| `funny_words` hardcoded in `word_structures.py` | Loaded from external `funny_words.json` |
| Style guide hardcoded in `system_prompt.py` | Loaded from external `preferences.json` |
| 100 examples in `known100.csv` inside repo | Loaded from external CSV (thousands of rows) |
| `OutputCapture` dumps files + does extraction | Clean `OutputParser` returns structured data |
| `RLVRTemplateTags` duplicated in 2 files | Single `RLVRConfig` dataclass |
| Google Drive integration | Removed (out of scope) |
| No Hub push | `datasets` library push to HF Hub |

---

## Config Injection Pattern

### Recommendation: Config Objects via Factory Function

Do not use dependency injection frameworks. Do not use environment variables for structured data. Use a simple factory function that loads files and returns typed config objects.

```python
# config.py

@dataclass(frozen=True)
class AppConfig:
    funny_words: list[str]
    preferences: str          # opaque text blob, injected into prompt verbatim
    human_examples: list[dict] # [{input: str, output: str}, ...]
    llm_backend: str           # "cerebras/qwen-3-32b", "openai/gpt-4o", etc.
    llm_api_key: str
    output_format: str         # "grpo", "dpo", "sft"

def load_config(
    funny_words_path: str,
    preferences_path: str,
    examples_csv_path: str,
    llm_backend: str = "cerebras/qwen-3-32b",
) -> AppConfig:
    """Load all external config files and return a single config object."""
    with open(funny_words_path) as f:
        funny_words = json.load(f)

    with open(preferences_path) as f:
        preferences = json.load(f)["style_description"]

    human_examples = []
    with open(examples_csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            human_examples.append({"input": row["input"], "output": row["output"]})

    return AppConfig(
        funny_words=funny_words,
        preferences=preferences,
        human_examples=human_examples,
        llm_backend=llm_backend,
        llm_api_key=os.environ.get("LLM_API_KEY", ""),
        output_format="grpo",
    )
```

### Why This Pattern

1. **Frozen dataclass** -- immutable after creation, safe to pass around, no accidental mutation
2. **Single load point** -- all file I/O happens in one place, easy to test (mock the function)
3. **Typed fields** -- IDE autocompletion and type checking catch errors early
4. **Environment vars only for secrets** -- API keys come from env vars, structured data from files
5. **No framework overhead** -- no DI container, no registry, just a function that returns data

### Config File Formats

**`funny_words.json`** -- flat list:
```json
["fart", "poop", "butt", "taco", "wiener", "pickle", "monkey", "splat"]
```

**`preferences.json`** -- opaque style description:
```json
{
  "style_description": "I like edgy, adult humor. Push boundaries. Food puns are always good. Bodily function humor is great. Think South Park meets Weird Al.",
  "phonetic_threshold": 0.6
}
```

The app injects `style_description` into the prompt as-is. It never parses the text.

**`human_examples.csv`** -- input/output pairs:
```csv
input,output
The Matrix,The Mattress
Die Hard,Dye Hard
Star Wars,Scar Whores
```

### What NOT to Do

- Do NOT load config inside individual modules. Pass the config object down from the entry point.
- Do NOT use `os.environ` for funny words or preferences. These are structured data, not secrets.
- Do NOT parse `preferences.json` beyond extracting the text blob. The whole point is opacity.
- Do NOT store config files in the repo. They are user-specific runtime inputs.

---

## LLM Adapter Layer

### Recommendation: Use smolagents Built-in Models

The current `CerebrasModel` reimplements what smolagents already provides. As of smolagents v1.24.0, the library offers:

| Class | Use Case |
|---|---|
| `LiteLLMModel` | 100+ providers via LiteLLM (Cerebras, OpenAI, Anthropic, etc.) |
| `InferenceClientModel` | HuggingFace Inference Providers (including Cerebras) |
| `OpenAIModel` | Any OpenAI-compatible endpoint |
| `TransformersModel` | Local HuggingFace models |

**Key finding: Cerebras is already a supported provider in `InferenceClientModel`.** And `LiteLLMModel` supports `cerebras/` prefixed model IDs.

### Implementation

```python
# llm.py

from smolagents import LiteLLMModel, InferenceClientModel

def create_model(backend: str, api_key: str, **kwargs) -> "Model":
    """
    Create a smolagents-compatible model from a backend string.

    backend format: "provider/model_name"
    Examples:
        "cerebras/qwen-3-32b"
        "openai/gpt-4o"
        "anthropic/claude-4-sonnet"
        "huggingface/Qwen/Qwen3-32B"
    """
    if backend.startswith("huggingface/"):
        model_id = backend.removeprefix("huggingface/")
        return InferenceClientModel(
            model_id=model_id,
            token=api_key,
            **kwargs,
        )
    else:
        # LiteLLM handles cerebras/, openai/, anthropic/, etc.
        return LiteLLMModel(
            model_id=backend,
            api_key=api_key,
            **kwargs,
        )
```

### Why LiteLLMModel Over Custom Adapter

1. **No `_preprocess_content` hack needed** -- LiteLLM handles message formatting per-provider
2. **No `ModelResponse` wrapper** -- smolagents expects `ChatMessage` objects, which built-in models return
3. **Built-in rate limiting** -- `requests_per_minute` parameter
4. **Built-in retry** -- exponential backoff on rate limit errors
5. **Provider switching is a string change** -- `"cerebras/qwen-3-32b"` to `"openai/gpt-4o"`

### Interaction with CodeAgent

The smolagents `CodeAgent` interface remains the same regardless of model backend:

```python
from smolagents import CodeAgent, load_tool

model = create_model(config.llm_backend, config.llm_api_key, temperature=0.7)
word_phone_tool = load_tool("patruff/word-phone", trust_remote_code=True)

agent = CodeAgent(
    tools=[word_phone_tool],
    model=model,
    system_prompt=system_prompt,
    additional_authorized_imports=["json"],
    step_callbacks=[capture.callback],
)

result = agent.run(prompt)
```

### Important: smolagents Model Interface

Custom models must subclass `smolagents.Model` and implement `generate()`:

```python
def generate(
    self,
    messages: list,
    stop_sequences: list[str] | None = None,
    response_format: dict[str, str] | None = None,
    tools_to_call_from: list[Tool] | None = None,
    **kwargs,
) -> ChatMessage:
```

The current `CerebrasModel.__call__` returns a `ModelResponse(content=...)`. This is outdated -- smolagents v1.24.0 uses `generate()` returning `ChatMessage`. This is another reason to use built-in models.

---

## Data Flow

### Complete Pipeline: Input CSV to HuggingFace Hub

```
INPUT
  input.csv (title column)
  funny_words.json (external)
  preferences.json (external)
  human_examples.csv (external)
      |
      v
[1. CONFIG LOADING]
  load_config() -> AppConfig (frozen dataclass)
      |
      v
[2. MODEL INITIALIZATION]
  create_model(config.llm_backend, config.llm_api_key) -> smolagents Model
      |
      v
[3. FOR EACH TITLE IN CSV:]
  |
  |--[3a. PRE-COMPUTE SUGGESTIONS]
  |    parody_tool.forward(word, funny_words, min_similarity=0.6)
  |    -> {word: [{suggestion, score}, ...]}
  |
  |--[3b. BUILD PROMPT]
  |    Select N random human examples from human_examples.csv
  |    Inject preferences text verbatim
  |    Inject suggestions JSON
  |    -> complete prompt string
  |
  |--[3c. RUN AGENT]
  |    CodeAgent with word_phone_tool
  |    step_callbacks capture each reasoning step
  |    -> raw text output with <think> tags, tool calls, attempts
  |
  |--[3d. PARSE OUTPUT]
  |    Extract: thinking_trace, tool_calls, attempts[], final_parody
  |    Compute: avg_phonetic_score, all_scores_valid, tool_usage_count
  |    -> RLVRDataPoint (structured)
  |
  v
[4. COLLECT ALL DATA POINTS]
  List[RLVRDataPoint] for all titles
      |
      v
[5. CONVERT TO TRL FORMAT]
  |
  |-- For GRPO/RLVR (prompt-only):
  |   {"prompt": "Create a parody of 'The Matrix'...",
  |    "ground_truth": <for reward function>}
  |
  |-- For DPO (preference):
  |   {"prompt": ..., "chosen": ..., "rejected": ...}
  |
  |-- For SFT (prompt-completion):
  |   {"prompt": ..., "completion": ...}
  |
      v
[6. PUSH TO HUB]
  datasets.Dataset.from_list(converted_data)
  dataset.push_to_hub("patruff/chuckles-rlvr-v1")
```

### TRL Dataset Format Details

**For GRPO (the primary RLVR target):**

GRPOTrainer expects **prompt-only** format. The model generates completions online. Rewards are computed by custom reward functions.

```python
# What we push to HuggingFace Hub:
{
    "prompt": [
        {"role": "user", "content": "Create a funny parody of 'The Matrix'. Use phonetic similarity > 0.6."}
    ],
    # Additional columns passed to reward functions as kwargs:
    "original_title": "The Matrix",
    "known_good_parody": "The Mattress",  # from human examples, if available
}
```

**Reward functions** (run during training, not during data generation):
```python
def phonetic_reward(completions, original_title, **kwargs) -> list[float]:
    """Check if the parody words pass phonetic similarity > 0.6"""
    ...

def format_reward(completions, **kwargs) -> list[float]:
    """Check if output has <think>...</think> reasoning tags"""
    ...

def humor_reward(completions, known_good_parody, **kwargs) -> list[float]:
    """Optional: compare to known good parody via embedding similarity"""
    ...
```

**Critical insight:** For RLVR/GRPO, we do NOT include the model's response in the dataset. The dataset is prompts only. The model generates responses during training, and reward functions score them. What we DO include are the metadata columns that reward functions need (original_title, known_good_parody).

**However**, we also want to capture the generation-time reasoning traces for analysis. So we produce TWO outputs:

1. **GRPO training dataset** (prompt-only + metadata) -- pushed to Hub
2. **Reasoning trace archive** (full structured data) -- saved as JSONL for analysis

### Output Schema for Reasoning Archive

```python
@dataclass
class GenerationRecord:
    """One generation run for one title. Saved to JSONL archive."""
    input_title: str
    prompt_used: str           # full prompt sent to agent
    raw_output: str            # complete agent output
    thinking_trace: str        # content between <think> tags
    tool_calls: list[dict]     # [{tool, args, result}, ...]
    attempts: list[dict]       # [{parody_text, phonetic_checks, humor_rating}, ...]
    final_parody: str
    final_reasoning: str

    # Quality signals
    avg_phonetic_score: float
    all_scores_valid: bool
    tool_usage_count: int

    # Metadata
    model_backend: str
    timestamp: str
    config_hash: str           # hash of config used, for reproducibility
```

---

## Module Structure

### Recommended Package Layout

```
chucklesPRIME/
|-- pyproject.toml              # Package metadata, dependencies
|-- README.md                   # Usage documentation
|
|-- src/
|   |-- chuckles/
|   |   |-- __init__.py
|   |   |
|   |   |-- config.py           # AppConfig dataclass + load_config()
|   |   |-- llm.py              # create_model() factory
|   |   |-- prompt.py           # PromptBuilder: assembles agent prompts
|   |   |-- agent.py            # run_agent(): CodeAgent setup + execution
|   |   |-- parser.py           # OutputParser: extracts structured data from raw output
|   |   |-- dataset.py          # RLVR/DPO/SFT format conversion + Hub push
|   |   |-- pipeline.py         # Main orchestrator: CSV in -> process -> dataset out
|   |   |
|   |   |-- tools/              # Phonetic tools (reference copies, deployed on HF Hub)
|   |   |   |-- __init__.py
|   |   |   |-- word_phone.py
|   |   |   |-- parody_suggestions.py
|   |   |
|   |   |-- prompts/
|   |   |   |-- __init__.py
|   |   |   |-- system.py       # AGENT_SYSTEM_PROMPT, PARODY_STYLE_GUIDE
|   |   |   |-- templates.py    # GENERATION_PROMPT_TEMPLATE, RLVR tag configs
|   |
|-- cli.py                      # CLI entry point (argparse)
|
|-- tests/
|   |-- test_config.py
|   |-- test_prompt.py
|   |-- test_parser.py
|   |-- test_dataset.py
|   |-- test_pipeline.py        # Integration test (mocked LLM)
|   |-- fixtures/
|   |   |-- sample_output.txt   # Real agent output for parser tests
|   |   |-- sample_config/
|   |       |-- funny_words.json
|   |       |-- preferences.json
|   |       |-- examples.csv
```

### Module Responsibilities

| Module | Responsibility | Imports From | Imported By |
|---|---|---|---|
| `config.py` | Load external files, return `AppConfig` | stdlib only | `pipeline.py`, `cli.py` |
| `llm.py` | Create smolagents model from backend string | `smolagents` | `pipeline.py` |
| `prompt.py` | Build complete prompt from title + config data | `prompts/` | `pipeline.py` |
| `agent.py` | Initialize CodeAgent, run it, return raw output | `smolagents`, `llm.py` | `pipeline.py` |
| `parser.py` | Extract structured data from raw agent output | stdlib (re, json) | `pipeline.py` |
| `dataset.py` | Convert parsed data to TRL formats, push to Hub | `datasets`, `huggingface_hub` | `pipeline.py` |
| `pipeline.py` | Orchestrate: CSV -> agent -> parse -> dataset | all above | `cli.py` |
| `cli.py` | Parse args, call pipeline | `pipeline.py`, `config.py` | (entry point) |

### Key Design Decisions

1. **`src/` layout with `pyproject.toml`** -- standard Python packaging. Avoids import confusion. `pip install -e .` for development.

2. **Tools stay on HuggingFace Hub** -- `word_phone.py` and `parody_suggestions.py` in `tools/` are reference copies only. The agent loads them via `load_tool("patruff/word-phone")`. Keeping local copies enables testing without network.

3. **`prompts/` is a sub-package** -- prompt templates are large strings. Separating them keeps other modules readable. `system.py` has the agent system prompt. `templates.py` has the generation template and RLVR tag configuration.

4. **`parser.py` is pure functions** -- no state, no I/O. Takes a string, returns a dataclass. Easy to unit test with fixture files.

5. **`pipeline.py` is the only module that knows about CSV** -- other modules work with Python objects. File I/O is isolated to pipeline and config.

---

## Human Examples Integration

### How Human Examples Flow Into Generation

Human examples serve two purposes:
1. **Few-shot examples in the prompt** -- show the model what good parodies look like
2. **Ground truth for RLVR reward functions** -- if we have a known parody for a title, the reward function can compare

### Prompt Integration Strategy

The `PromptBuilder` selects a subset of human examples for each generation:

```python
class PromptBuilder:
    def __init__(self, config: AppConfig):
        self.examples = config.human_examples
        self.preferences = config.preferences

    def build(self, title: str, suggestions: dict, num_examples: int = 15) -> str:
        # 1. Select examples
        #    - If we have a known parody for this exact title, EXCLUDE it
        #      (don't give away the answer)
        #    - Pick num_examples random examples from the pool
        #    - Bias toward examples with similar word count to the input title
        selected = self._select_examples(title, num_examples)

        # 2. Format examples section
        examples_text = "\n".join(
            f'  - "{ex["input"]}" -> "{ex["output"]}"'
            for ex in selected
        )

        # 3. Inject preferences verbatim
        # 4. Inject suggestions
        # 5. Return completed template
        return GENERATION_PROMPT_TEMPLATE.format(
            title=title,
            examples_text=examples_text,
            style_preferences=self.preferences,  # opaque text blob
            suggestions=json.dumps(suggestions, indent=2),
        )
```

### Scaling Considerations for Thousands of Examples

With thousands of examples, we cannot put them all in the prompt. Strategies:

1. **Random sampling** (baseline) -- pick 10-20 random examples per generation. Simple, works.
2. **Similarity-based selection** -- pick examples whose input titles are structurally similar (same word count, similar genres). Requires a lightweight similarity metric.
3. **Category balancing** -- ensure selected examples cover different parody strategies (single word swap, double swap, compound word, etc.).

For v1, random sampling with exclusion of the target title is sufficient.

### Ground Truth for Reward Functions

When pushing the GRPO dataset, include a `known_good_parody` column if one exists:

```python
for title in titles:
    row = {"prompt": build_prompt(title), "original_title": title}

    # Check if we have a human example for this title
    matching = [ex for ex in config.human_examples if ex["input"] == title]
    if matching:
        row["known_good_parody"] = matching[0]["output"]
    else:
        row["known_good_parody"] = ""

    dataset_rows.append(row)
```

---

## Build Order

### Dependency Graph

```
config.py          (no internal deps)
    |
    v
llm.py             (depends on: smolagents)
    |
    v
prompts/system.py  (no internal deps)
prompts/templates.py (no internal deps)
    |
    v
prompt.py          (depends on: config, prompts/)
    |
    v
agent.py           (depends on: llm, smolagents)
    |
    v
parser.py          (no internal deps -- pure functions)
    |
    v
dataset.py         (depends on: parser output types, datasets library)
    |
    v
pipeline.py        (depends on: all above)
    |
    v
cli.py             (depends on: pipeline, config)
```

### Suggested Build Phases

**Phase 1: Foundation (no LLM calls needed)**

Build order:
1. `config.py` + `AppConfig` dataclass + `load_config()` factory
2. `parser.py` + `GenerationRecord` dataclass + extraction functions
3. `prompts/system.py` + `prompts/templates.py` (port from current `system_prompt.py`)
4. Tests for config loading (with fixture files)
5. Tests for parser (with fixture sample output from current system)

Why first: These modules have zero external dependencies. You can test them immediately with fixture data from the existing `parodies2026/` output. The parser is the most fragile component (regex-based) and benefits from early testing.

**Phase 2: LLM + Agent (requires API key)**

Build order:
5. `llm.py` + `create_model()` factory
6. `prompt.py` + `PromptBuilder` class
7. `agent.py` + `run_agent()` function
8. Manual smoke test: single title generation with new structure

Why second: Once config and parsing work, you can wire up the LLM. The critical validation is that `LiteLLMModel` or `InferenceClientModel` works as a drop-in replacement for `CerebrasModel` with the existing smolagents CodeAgent.

**Phase 3: Dataset + Hub (requires HuggingFace token)**

Build order:
9. `dataset.py` + format converters (GRPO prompt-only, DPO preference, SFT)
10. Hub push function using `datasets` library
11. Tests for format conversion (unit tests, no network)
12. Integration test: push small test dataset to Hub

Why third: This is the output end. It depends on having real generation records to convert. Build after the generation pipeline works.

**Phase 4: Pipeline + CLI (integration)**

Build order:
13. `pipeline.py` + main orchestration loop
14. `cli.py` + argparse entry point
15. End-to-end test: `input.csv` -> generation -> RLVR dataset -> Hub push
16. `pyproject.toml` + packaging

Why last: The pipeline is pure glue code. It should be trivial once all components work independently.

### Risk Mitigation Per Phase

| Phase | Risk | Mitigation |
|---|---|---|
| 1 | Parser regex doesn't match new model output | Collect 5+ real outputs from current system as test fixtures |
| 2 | `LiteLLMModel` doesn't work with Cerebras | Fall back to `InferenceClientModel` with `provider="cerebras"`, or keep thin custom adapter as escape hatch |
| 2 | smolagents CodeAgent API changed between versions | Pin `smolagents>=1.24.0,<2.0.0` |
| 3 | TRL dataset format requirements are unclear | Use `datasets` library directly; format is just a dict with "prompt" key; additional columns passed to reward functions |
| 4 | Batch processing too slow | Irrelevant for v1 (correctness first), but `asyncio` or process pool can be added to pipeline.py later |

### What to Port vs. Rewrite

| File | Action | Reason |
|---|---|---|
| `word_phone.py` | Keep as-is on Hub | Works, deployed, used by other projects |
| `parody_suggestions.py` | Keep as-is on Hub | Works, deployed |
| `system_prompt.py` | Port to `prompts/` | Good content, needs restructuring only |
| `word_structures.py` | Extract data to external JSON | The code is just data holders; move `funny_words` to JSON, `custom_phones` to JSON, `known_parodies` CSV stays external |
| `generate_parody.py` | Rewrite as `agent.py` + `prompt.py` | Too many responsibilities in one file (model adapter, agent setup, output capture, file I/O, CLI) |
| `rlvr_dataset_tools.py` | Port to `dataset.py` | Good conversion logic, remove interactive labeler (out of scope), update to TRL format |
| `test_popular_movies.py` | Port `EnhancedOutputCapture` to `parser.py` | Good extraction logic, decouple from test harness |
| `batch_generate.py` | Replace with `pipeline.py` | Simplified; no more Google Drive |
| `drive_batch_processor.py` | Delete | Out of scope |
| `upload_to_drive.py` | Delete | Out of scope |
| `push_tool_to_hub.py` | Delete (one-time utility) | Tools already on Hub |

---

## Appendix: Key Reference Links

- [smolagents Models Reference](https://huggingface.co/docs/smolagents/en/reference/models) -- all built-in model classes
- [TRL Dataset Formats](https://huggingface.co/docs/trl/main/en/dataset_formats) -- prompt-only, preference, etc.
- [TRL GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) -- RLVR via GRPO
- [TRL Reward Functions](https://huggingface.co/docs/trl/en/rewards) -- built-in rewards (accuracy, format, reasoning)
- [LiteLLM Providers](https://docs.litellm.ai/docs/) -- 100+ supported LLM providers
- [TRL RLVR Trainer Proposal](https://github.com/huggingface/trl/issues/4711) -- dedicated RLVR trainer (December 2025)

---

*Architecture research: 2026-01-31*
