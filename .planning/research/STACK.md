# Stack Research

> Research date: 2026-01-31
> Target: chucklesPRIME -- RLVR data generation pipeline for phonetic parody titles

---

## Recommended Stack

| Library | Version | Role | Rationale |
|---------|---------|------|-----------|
| `smolagents` | `>=1.24.0` | Agent orchestration | HuggingFace's official lightweight agent framework. Successor to `transformers.agents`. ~1000 LOC core, first-class CodeAgent support, swappable model backends. |
| `datasets` | `>=4.5.0` | Dataset creation & upload | HuggingFace's standard dataset library. Native `push_to_hub`, Parquet serialization, Dataset Viewer support. |
| `huggingface-hub` | `>=1.3.5` | Hub authentication & API | Client library for all Hub interactions. Handles login, token management, push/pull. Dependency of both smolagents and datasets. |
| `trl` | `>=0.27.0` | RLVR format reference | Defines the GRPOTrainer dataset contract. We generate data *for* TRL, not run TRL itself. Pin loosely -- we only need the format spec. |
| `pronouncing` | `>=0.2.0` | Phonetic analysis | Simple Python interface to CMU Pronouncing Dictionary. Provides `phones_for_word()`, `rhymes()`, `stresses()`, `search()`. BSD license, no heavy deps. |
| `cmudict` | `>=1.1.3` | Phonetic dictionary data | Versioned wrapper for CMU dict data files. Auto-installed as dependency of `pronouncing`. |
| `litellm` | `>=1.55.0` | Multi-provider LLM routing | Required by `smolagents[litellm]` for `LiteLLMModel`. Supports 100+ providers including Cerebras, OpenAI, Anthropic. |
| `openai` | `>=1.50.0` | OpenAI-compatible API client | Required by `smolagents[openai]` for `OpenAIModel`. Used when pointing at Cerebras or other OpenAI-compatible endpoints. |
| `python-dotenv` | `>=1.0.0` | Environment variable management | Load `.env` files for API keys. Standard practice for secrets management. |

**Python version**: `>=3.10` (smolagents and datasets both require `>=3.9`; 3.10+ for match/case and modern typing)

**Runtime note**: This is a *data generation* pipeline, not a training pipeline. We do NOT need `torch`, `transformers`, `accelerate`, or GPU resources. All LLM calls go through remote APIs.

---

## smolagents Integration

### Current API (v1.24.0)

smolagents provides two agent paradigms and multiple model backends. The library is the official HuggingFace successor to `transformers.agents`.

### Agent Classes

```python
from smolagents import CodeAgent, ToolCallingAgent
```

| Class | How it works | Best for |
|-------|-------------|----------|
| `CodeAgent` | Generates Python code snippets that call tools as functions | Complex multi-step reasoning, chaining tool outputs |
| `ToolCallingAgent` | Outputs structured JSON tool calls (OpenAI function-calling style) | Simpler single-tool invocations |

Both inherit from `MultiStepAgent` and operate in a loop of **Thought -> Action -> Observation** steps.

**Recommendation for chucklesPRIME**: Use `CodeAgent`. It produces richer reasoning traces (the code *is* the reasoning) and naturally chains phonetic lookup -> word selection -> title construction.

### Tool Creation

**Option 1: `@tool` decorator (recommended for our use case)**

```python
from smolagents import tool

@tool
def get_phonemes(word: str) -> str:
    """
    Returns the ARPAbet phoneme string for a given English word.

    Args:
        word: The English word to look up phonemes for.
    """
    import pronouncing
    phones = pronouncing.phones_for_word(word.lower())
    if not phones:
        return f"No phonemes found for '{word}'"
    return phones[0]

@tool
def find_rhymes(word: str) -> list[str]:
    """
    Finds all words that rhyme with the given word.

    Args:
        word: The English word to find rhymes for.
    """
    import pronouncing
    return pronouncing.rhymes(word.lower())

@tool
def get_stress_pattern(word: str) -> str:
    """
    Returns the stress pattern (digits only) for a word's pronunciation.

    Args:
        word: The English word to analyze.
    """
    import pronouncing
    phones = pronouncing.phones_for_word(word.lower())
    if not phones:
        return ""
    return pronouncing.stresses(phones[0])
```

Key requirements for `@tool` decorated functions:
- **Type hints** on all parameters and return type (mandatory)
- **Docstring** with `Args:` section describing each parameter (mandatory -- the LLM reads this)
- **Descriptive function name** (the LLM uses the name to decide when to call it)

**Option 2: Subclass `Tool` (for stateful tools)**

```python
from smolagents import Tool

class FunnyWordLookupTool(Tool):
    name = "funny_word_lookup"
    description = "Looks up funny replacement words from the loaded configuration."
    inputs = {
        "category": {"type": "string", "description": "The category of funny words to search."}
    }
    output_type = "array"

    def __init__(self, funny_words: dict):
        super().__init__()
        self.funny_words = funny_words

    def forward(self, category: str) -> list[str]:
        return self.funny_words.get(category, [])
```

Use the subclass pattern when you need to inject runtime config (like the funny words JSON) as instance state.

### Model Classes (v1.24.0)

smolagents provides these model adapters, all sharing the same `generate()` interface:

| Class | Backend | When to use |
|-------|---------|-------------|
| `InferenceClientModel` | HuggingFace Inference Providers (Cerebras, Together, Fireworks, etc.) | Default choice. Free tier available. Supports `provider="cerebras"`. Replaces the older `HfApiModel`. |
| `LiteLLMModel` | 100+ providers via LiteLLM | When you need Cerebras (`cerebras/llama-3.3-70b`), Anthropic, or other providers not on HF Inference. |
| `OpenAIModel` | OpenAI API or any compatible server | Direct OpenAI usage, or pointing `api_base` at Cerebras (`https://api.cerebras.ai/v1`), vLLM, etc. |
| `LiteLLMRouterModel` | Multiple providers with load balancing | Failover across Cerebras + Together + Groq, etc. |
| `TransformersModel` | Local HuggingFace models | Local inference (requires torch). Not recommended for our API-only pipeline. |
| `AzureOpenAIModel` | Azure OpenAI | Azure deployments. |
| `AmazonBedrockModel` | AWS Bedrock | AWS Bedrock models. |
| `MLXModel` | Apple Silicon MLX | Local on Mac. |
| `VLLMModel` | vLLM server | High-throughput local serving. |

**All model classes** accept `temperature`, `max_tokens`, `top_p`, etc. at instantiation time.

### Agent Instantiation Pattern

```python
from smolagents import CodeAgent, InferenceClientModel

model = InferenceClientModel(
    model_id="Qwen/Qwen2.5-72B-Instruct",
    provider="cerebras",
    temperature=0.7,
    max_tokens=2048,
)

agent = CodeAgent(
    model=model,
    tools=[get_phonemes, find_rhymes, get_stress_pattern, funny_word_lookup],
    max_steps=6,
)

result = agent.run("Create a funny parody of 'The Great Gatsby'")
```

### Capturing Reasoning Traces

This is critical for RLVR dataset generation. After `agent.run()`:

```python
# Fine-grained step-by-step logs (list of dicts, one per step)
trace_logs = agent.logs

# Structured chat-message format of the reasoning
memory_messages = agent.write_inner_memory_from_logs()

# Direct access to memory step objects
steps = agent.memory.steps
```

Each step in `agent.logs` contains:
- The LLM's **thought** (reasoning text)
- The **code** generated (for CodeAgent) or **tool call** JSON (for ToolCallingAgent)
- The **observation** (tool execution result)
- Timing and token usage metadata

**This is exactly what we need** to build the `reasoning_trace` field for our RLVR dataset.

### Multi-Agent Orchestration

For more complex pipelines, agents can manage other agents:

```python
from smolagents import CodeAgent

phonetic_agent = CodeAgent(
    model=model,
    tools=[get_phonemes, find_rhymes],
    name="phonetic_analyzer",
    description="Analyzes phonetic properties of words and finds rhymes.",
)

manager_agent = CodeAgent(
    model=model,
    tools=[funny_word_lookup],
    managed_agents=[phonetic_agent],
)
```

The manager agent's system prompt automatically includes descriptions of managed agents so it knows how to delegate.

---

## RLVR Dataset Format

### What GRPOTrainer Expects

The TRL `GRPOTrainer` (Group Relative Policy Optimization) is the standard trainer for RLVR. It expects a **prompt-only** dataset with one required column and optional auxiliary columns.

### Required Column

| Column | Type | Format Options |
|--------|------|---------------|
| `prompt` | `str` or `list[dict]` | **Standard**: plain text string. **Conversational**: list of `{"role": "...", "content": "..."}` message dicts. |

### Auxiliary Columns (Passed to Reward Functions)

Any additional columns are passed to reward functions as keyword arguments. Common patterns:

| Column | Purpose | Example |
|--------|---------|---------|
| `answer` / `solution` | Ground truth for verification | `"The Grape Fatsby"` |
| `task` | Identifies which reward function to apply | `"phonetic_parody"` |
| `original_title` | Source material for the parody | `"The Great Gatsby"` |
| `reasoning_trace` | Chain-of-thought from generation | JSON string of agent steps |
| `phonetic_score` | Pre-computed phonetic similarity | `0.85` |
| `metadata` | Any structured metadata | JSON string |

### chucklesPRIME Target Dataset Schema

For our specific use case, the dataset should look like:

```python
{
    # REQUIRED by GRPOTrainer
    "prompt": [
        {"role": "system", "content": "You are a comedy writer who creates funny parody titles..."},
        {"role": "user", "content": "Create a phonetically similar parody of the title 'The Great Gatsby'"}
    ],

    # AUXILIARY -- passed to reward functions via **kwargs
    "original_title": "The Great Gatsby",
    "parody_title": "The Grape Fatsby",
    "reasoning_trace": "[{\"thought\": \"I need to find words that sound like...\", \"code\": \"...\", \"observation\": \"...\"}]",
    "phonetic_distance": 0.15,
    "funny_words_used": ["grape", "fatsby"],
    "generation_model": "Qwen/Qwen2.5-72B-Instruct",
    "timestamp": "2026-01-31T12:00:00Z"
}
```

### Standard vs Conversational Format

**Conversational format is strongly recommended** because:
1. GRPOTrainer applies the model's chat template automatically
2. It preserves system prompt instructions
3. It matches how the model was instruction-tuned

```python
# GOOD: Conversational format
{"prompt": [
    {"role": "system", "content": "You create phonetic parody titles."},
    {"role": "user", "content": "Make a parody of 'War and Peace'"}
]}

# ALSO VALID: Standard format (simpler but less control)
{"prompt": "Create a funny phonetic parody of the title 'War and Peace'"}
```

### Reward Function Pattern for chucklesPRIME

When someone trains with our dataset, they would write reward functions like:

```python
def phonetic_similarity_reward(completions, original_title, **kwargs):
    """Reward based on phonetic similarity between parody and original."""
    import pronouncing
    rewards = []
    for completion, orig in zip(completions, original_title):
        # Extract the parody title from the completion
        parody = extract_title(completion)
        score = compute_phonetic_similarity(parody, orig)
        rewards.append(score)
    return rewards

def humor_format_reward(completions, **kwargs):
    """Reward for proper formatting (has a title, is funny-sounding)."""
    rewards = []
    for completion in completions:
        has_title = bool(re.search(r'"[^"]+"|\'[^\']+\'', completion))
        rewards.append(1.0 if has_title else 0.0)
    return rewards

# GRPOTrainer usage with our dataset
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=[phonetic_similarity_reward, humor_format_reward],
    train_dataset=our_dataset,  # chucklesPRIME output
    args=GRPOConfig(remove_unused_columns=False),  # CRITICAL: keeps auxiliary columns
)
```

**Key config**: `remove_unused_columns=False` must be set in `GRPOConfig` to preserve auxiliary columns for reward functions.

---

## Dataset Upload

### Creating and Pushing a Dataset

```python
from datasets import Dataset, Features, Value, Sequence
from huggingface_hub import login

# Authenticate (token from env or interactive prompt)
login(token=os.environ.get("HF_TOKEN"))

# Define schema explicitly (recommended for reproducibility)
features = Features({
    "prompt": [{  # Conversational format = list of dicts
        "role": Value("string"),
        "content": Value("string"),
    }],
    "original_title": Value("string"),
    "parody_title": Value("string"),
    "reasoning_trace": Value("string"),  # JSON-serialized
    "phonetic_distance": Value("float32"),
    "funny_words_used": Sequence(Value("string")),
    "generation_model": Value("string"),
    "timestamp": Value("string"),
})

# Build from list of dicts
records = [
    {
        "prompt": [
            {"role": "system", "content": "You create phonetic parodies."},
            {"role": "user", "content": "Make a parody of 'The Great Gatsby'"},
        ],
        "original_title": "The Great Gatsby",
        "parody_title": "The Grape Fatsby",
        "reasoning_trace": json.dumps(agent.logs),
        "phonetic_distance": 0.15,
        "funny_words_used": ["grape", "fatsby"],
        "generation_model": "Qwen/Qwen2.5-72B-Instruct",
        "timestamp": datetime.now().isoformat(),
    },
    # ... more records
]

dataset = Dataset.from_list(records)

# Push to Hub
dataset.push_to_hub(
    "username/chuckles-rlvr-dataset",
    private=True,           # Start private, make public later
)
```

### Incremental / Append Pattern

For long-running generation pipelines that produce data over time:

```python
# Save locally as you generate
dataset.save_to_disk("/path/to/local/chuckles_dataset")

# Or push periodically with split names
dataset.push_to_hub(
    "username/chuckles-rlvr-dataset",
    split="train",
    private=True,
)
```

### From Pandas (if collecting in a DataFrame)

```python
import pandas as pd
from datasets import Dataset

df = pd.DataFrame(records)
dataset = Dataset.from_pandas(df)
dataset.push_to_hub("username/chuckles-rlvr-dataset")
```

### Authentication Best Practices

1. **Environment variable** (preferred): Set `HF_TOKEN` in `.env` file, load with `python-dotenv`
2. **CLI login**: Run `hf auth login` once per machine
3. **Programmatic**: `login(token=os.environ["HF_TOKEN"])` at script start
4. **Never** hardcode tokens in source code

```python
# In your pipeline entry point:
from dotenv import load_dotenv
from huggingface_hub import login
import os

load_dotenv()  # loads .env file
login(token=os.environ.get("HF_TOKEN"))
```

---

## LLM Backend Flexibility

### Architecture: Factory Pattern with smolagents Model Classes

The recommended pattern for swappable backends:

```python
from smolagents import (
    InferenceClientModel,
    LiteLLMModel,
    OpenAIModel,
)

def create_model(config: dict):
    """Factory function to create the appropriate model from config."""
    backend = config["backend"]

    if backend == "huggingface":
        return InferenceClientModel(
            model_id=config["model_id"],
            provider=config.get("provider", "auto"),
            token=os.environ.get("HF_TOKEN"),
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 2048),
        )
    elif backend == "cerebras":
        # Option A: Via InferenceClientModel (if Cerebras is an HF Inference Provider)
        return InferenceClientModel(
            model_id=config["model_id"],
            provider="cerebras",
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 2048),
        )
        # Option B: Via LiteLLMModel
        # return LiteLLMModel(
        #     model_id=f"cerebras/{config['model_id']}",
        #     api_key=os.environ.get("CEREBRAS_API_KEY"),
        #     temperature=config.get("temperature", 0.7),
        #     max_tokens=config.get("max_tokens", 2048),
        # )
        # Option C: Via OpenAIModel (OpenAI-compatible endpoint)
        # return OpenAIModel(
        #     model_id=config["model_id"],
        #     api_base="https://api.cerebras.ai/v1",
        #     api_key=os.environ.get("CEREBRAS_API_KEY"),
        #     temperature=config.get("temperature", 0.7),
        #     max_tokens=config.get("max_tokens", 2048),
        # )
    elif backend == "openai":
        return OpenAIModel(
            model_id=config["model_id"],
            api_key=os.environ.get("OPENAI_API_KEY"),
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 2048),
        )
    elif backend == "litellm":
        # Catch-all for any LiteLLM-supported provider
        return LiteLLMModel(
            model_id=config["model_id"],
            api_key=os.environ.get(config.get("api_key_env", "LLM_API_KEY")),
            api_base=config.get("api_base"),
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 2048),
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")
```

### Config-Driven Backend Selection (JSON)

```json
{
    "llm": {
        "backend": "cerebras",
        "model_id": "llama-3.3-70b",
        "provider": "cerebras",
        "temperature": 0.7,
        "max_tokens": 2048
    }
}
```

### Cerebras-Specific Notes

Cerebras can be accessed via **three** smolagents model classes:

| Route | Class | Config |
|-------|-------|--------|
| HF Inference Providers | `InferenceClientModel` | `provider="cerebras"` |
| LiteLLM | `LiteLLMModel` | `model_id="cerebras/llama-3.3-70b"` |
| OpenAI-compatible | `OpenAIModel` | `api_base="https://api.cerebras.ai/v1"` |

**Recommendation**: Start with `InferenceClientModel(provider="cerebras")` for simplicity. Fall back to `LiteLLMModel` if you need provider-specific features or routing.

### Resilience: LiteLLMRouterModel for Multi-Provider Failover

```python
from smolagents import LiteLLMRouterModel

model = LiteLLMRouterModel(
    model_id="llama-3.3-70b",
    model_list=[
        {
            "model_name": "llama-3.3-70b",
            "litellm_params": {
                "model": "cerebras/llama-3.3-70b",
                "api_key": os.getenv("CEREBRAS_API_KEY"),
            },
        },
        {
            "model_name": "llama-3.3-70b",
            "litellm_params": {
                "model": "together_ai/meta-llama/Llama-3.3-70B-Instruct",
                "api_key": os.getenv("TOGETHER_API_KEY"),
            },
        },
    ],
    client_kwargs={"routing_strategy": "simple-shuffle"},
)
```

---

## Dependencies

### requirements.txt

```
# Core agent framework
smolagents>=1.24.0

# Extras for model backends (install what you need)
# smolagents[litellm]   -- for LiteLLMModel
# smolagents[openai]    -- for OpenAIModel

# LLM provider support
litellm>=1.55.0
openai>=1.50.0

# HuggingFace ecosystem
datasets>=4.5.0
huggingface-hub>=1.3.5

# Phonetic analysis
pronouncing>=0.2.0

# Configuration & environment
python-dotenv>=1.0.0

# TRL (only needed if validating dataset format locally, NOT for generation)
# trl>=0.27.0
```

### Minimal Install (just generation + upload)

```bash
pip install smolagents[openai,litellm] datasets huggingface-hub pronouncing python-dotenv
```

### What We Do NOT Need

| Library | Why not |
|---------|---------|
| `torch` | No local model inference. All LLM calls are remote API. |
| `transformers` | No tokenization or model loading needed. smolagents handles chat templates. |
| `accelerate` | No distributed training. |
| `trl` | We generate data *for* TRL, but don't run training. Format spec is documented above. |
| `peft` / `bitsandbytes` | Training-side only. |
| `vllm` | Server-side inference engine. Not needed for API calls. |

---

## Confidence Levels

| Recommendation | Confidence | Notes |
|---------------|------------|-------|
| smolagents as agent framework | **High** | Official HF library, actively maintained (v1.24.0 Jan 2026), clear API, direct HF ecosystem integration. |
| `CodeAgent` over `ToolCallingAgent` | **High** | Richer reasoning traces (code = reasoning), better for multi-step phonetic analysis chains. |
| `@tool` decorator for phonetic tools | **High** | Well-documented, simple, stateless tools are the natural fit. Use `Tool` subclass only for config-injected state. |
| TRL prompt-only format with auxiliary columns | **High** | Directly documented in TRL docs. GRPOTrainer explicitly supports this via `remove_unused_columns=False`. |
| Conversational prompt format | **High** | Better than standard format for instruction-tuned models. Preserves system prompt. |
| `InferenceClientModel` as primary model class | **High** | Newest model class in smolagents. Supports Cerebras as a provider directly. Replaces older `HfApiModel`. |
| `datasets` library for Hub upload | **High** | Standard, well-maintained, 46M+ monthly downloads. `push_to_hub` is the canonical pattern. |
| `pronouncing` for phonetics | **High** | Stable (v0.2.0), lightweight, BSD license, wraps CMU dict (134K+ entries). Standard choice for English phonetics in Python. |
| `LiteLLMModel` for multi-provider routing | **Medium-High** | Well-supported in smolagents, but LiteLLM occasionally has model mapping issues with newer providers. Pin version carefully. |
| `python-dotenv` for secrets | **High** | Industry standard. Simple, no magic. |
| Skipping `torch`/`transformers` | **High** | Correct for an API-only data generation pipeline. Dramatically simplifies install and avoids CUDA issues. |
| `agent.logs` for reasoning trace capture | **Medium-High** | Works well but the exact structure of log entries is not formally versioned. May change between smolagents releases. Write defensive serialization code. |
| JSON config for funny words / preferences | **High** | Simple, human-editable, no dependencies. Standard Python `json.load()`. |

---

## Sources

- [smolagents Documentation](https://huggingface.co/docs/smolagents/index)
- [smolagents Models Reference](https://huggingface.co/docs/smolagents/en/reference/models)
- [smolagents Tools Tutorial](https://huggingface.co/docs/smolagents/en/tutorials/tools)
- [smolagents GitHub](https://github.com/huggingface/smolagents)
- [smolagents PyPI](https://pypi.org/project/smolagents/) (v1.24.0, Jan 16, 2026)
- [smolagents Agent Memory](https://huggingface.co/docs/smolagents/en/tutorials/memory)
- [smolagents Reasoning Trace Capture (Issue #322)](https://github.com/huggingface/smolagents/issues/322)
- [TRL Dataset Formats](https://huggingface.co/docs/trl/main/en/dataset_formats)
- [TRL GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [TRL PyPI](https://pypi.org/project/trl/) (v0.27.1)
- [TRL Reward Functions](https://huggingface.co/docs/trl/en/rewards)
- [datasets PyPI](https://pypi.org/project/datasets/) (v4.5.0, Jan 14, 2026)
- [HuggingFace Hub Upload Guide](https://huggingface.co/docs/datasets/en/upload_dataset)
- [huggingface-hub PyPI](https://pypi.org/project/huggingface-hub/) (v1.3.5, Jan 29, 2026)
- [HuggingFace Hub Authentication](https://huggingface.co/docs/huggingface_hub/en/package_reference/authentication)
- [pronouncing Library Documentation](https://pronouncing.readthedocs.io/en/latest/)
- [pronouncing PyPI](https://pypi.org/project/pronouncing/) (v0.2.0)
- [cmudict PyPI](https://pypi.org/project/cmudict/) (v1.1.3, Jan 3, 2026)
- [Cerebras OpenAI Compatibility](https://inference-docs.cerebras.ai/resources/openai)
- [LiteLLM Cerebras Provider](https://docs.litellm.ai/docs/providers/cerebras)
- [HuggingFace Agents Course - Tools](https://huggingface.co/learn/agents-course/en/unit2/smolagents/tools)
