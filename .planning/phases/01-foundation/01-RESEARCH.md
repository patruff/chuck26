# Phase 1: Foundation - Research

**Researched:** 2026-01-31
**Domain:** Python package structure, config loading, LLM model adapter, CSV data cleaning
**Confidence:** HIGH

## Summary

Phase 1 establishes the installable Python package, external config loading, and LLM connectivity. Research covered five domains: (1) the smolagents model interface that the custom LLM adapter must implement, (2) pyproject.toml project structure, (3) CSV data cleaning strategy for the ~1,234 human parody examples, (4) config loading patterns using frozen dataclasses, and (5) the OpenAI Python client API for wrapping Cerebras and other OpenAI-compatible backends.

The existing codebase in `parodies2026/` has a working `CerebrasModel` adapter, but it uses the **old** smolagents interface (`__call__` returning a custom `ModelResponse` dataclass). smolagents v1.24+ requires subclassing `smolagents.Model` and implementing a `generate()` method that returns `smolagents.ChatMessage`. This is the most critical migration.

The human_parodies.csv has **four distinct format zones** that require zone-specific parsing. The target output is a uniform 3-column table: `input`, `output`, `explanation`.

**Primary recommendation:** Subclass `smolagents.Model`, implement `generate()` returning `ChatMessage`, use the `openai` Python client internally (since Cerebras and all OpenAI-compatible APIs use the same wire format). Load all config via a frozen dataclass with a `load_config()` factory function.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| smolagents | >=1.24.0 | Agent framework (CodeAgent orchestration) | Already in use, HuggingFace official |
| openai | >=1.0.0 | OpenAI-compatible API client | Standard client for any chat completions API; Cerebras, Together, etc. all support this wire format |
| Python | >=3.10 | Runtime | smolagents requires 3.10+ |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| rich | latest | Console output formatting | Already in use |
| pronouncing | latest | CMU phonetic dictionary | Already in use for phonetic tools |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| openai client | cerebras-cloud-sdk | cerebras SDK is Cerebras-only; openai client works with ANY OpenAI-compatible endpoint. User wants backend flexibility. |
| openai client | LiteLLM | User explicitly chose custom adapter over LiteLLM per PROJECT.md decisions |
| openai client | smolagents.OpenAIModel directly | Could work, but user wants custom adapter wrapping OpenAI client directly for full control |
| frozen dataclass | pydantic BaseModel | Pydantic adds a dependency for validation we don't need; stdlib dataclasses are sufficient |

**Installation (dependencies for pyproject.toml):**
```
smolagents>=1.24.0
openai>=1.0.0
rich
pronouncing
```

Note: `cerebras-cloud-sdk`, `google-api-python-client`, `google-auth`, `google-auth-oauthlib`, `google-auth-httplib2` from the old `requirements.txt` are all dropped. The `openai` client replaces `cerebras-cloud-sdk`. Google Drive integration is out of scope per PROJECT.md.

## Architecture Patterns

### Recommended Project Structure
```
chucklesPRIME/
├── pyproject.toml
├── src/
│   └── chuckles_prime/
│       ├── __init__.py           # Package version, public API
│       ├── config.py             # AppConfig frozen dataclass + load_config()
│       ├── model.py              # Custom LLM adapter (subclass smolagents.Model)
│       ├── prompts.py            # System prompts (from system_prompt.py)
│       ├── tools.py              # Phonetic tools loader
│       ├── generate.py           # Parody generation pipeline
│       ├── word_structures.py    # Funny words, custom phones, known parodies
│       └── cli.py                # CLI entry point
├── tests/
│   ├── test_config.py
│   ├── test_model.py
│   └── test_csv_cleaning.py
└── scripts/
    └── clean_csv.py              # One-time CSV cleaning script
```

Uses **src layout** per Python packaging best practices. Package name `chuckles_prime` (underscores, PEP 8). Distribution name `chucklesPRIME` in pyproject.toml.

### Pattern 1: Frozen Dataclass Config with Factory Function
**What:** A `@dataclass(frozen=True)` class `AppConfig` that holds all loaded config, with a `load_config(settings_path)` factory function.
**When to use:** Loading external JSON + CSV into a typed, immutable config object.
**Confidence:** HIGH -- standard Python pattern, verified from official docs.

```python
# Source: Python docs (dataclasses), standard pattern
from dataclasses import dataclass, field
from pathlib import Path
import json
import csv

@dataclass(frozen=True)
class AppConfig:
    """Immutable application configuration loaded from external files."""
    # LLM backend
    model_name: str
    api_base_url: str
    api_key_env_var: str  # Name of env var holding the API key

    # External data (loaded from files)
    funny_words: dict[str, list[str]]   # category -> words
    preferences_text: str                # Opaque text injected into prompts
    human_examples: list[tuple[str, str, str]]  # (input, output, explanation)

    # Paths (for reference)
    funny_words_path: Path
    preferences_path: Path
    human_examples_path: Path


def load_config(settings_path: str | Path) -> AppConfig:
    """Load all configuration from a settings file that points to external files.

    Settings JSON structure:
    {
        "funny_words_path": "/path/to/funny_words.json",
        "preferences_path": "/path/to/preferences.json",
        "human_examples_path": "/path/to/human_examples.csv",
        "model_name": "qwen-3-32b",
        "api_base_url": "https://api.cerebras.ai/v1",
        "api_key_env_var": "CEREBRAS_API_KEY"
    }
    """
    settings_path = Path(settings_path)
    with open(settings_path) as f:
        settings = json.load(f)

    # Load funny words
    funny_words_path = Path(settings["funny_words_path"])
    with open(funny_words_path) as f:
        funny_words = json.load(f)

    # Load preferences (opaque text)
    preferences_path = Path(settings["preferences_path"])
    with open(preferences_path) as f:
        prefs = json.load(f)
    preferences_text = prefs.get("style_description", "")

    # Load human examples from cleaned CSV
    human_examples_path = Path(settings["human_examples_path"])
    human_examples = []
    with open(human_examples_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            human_examples.append((row["input"], row["output"], row["explanation"]))

    return AppConfig(
        model_name=settings["model_name"],
        api_base_url=settings["api_base_url"],
        api_key_env_var=settings["api_key_env_var"],
        funny_words=funny_words,
        preferences_text=preferences_text,
        human_examples=human_examples,
        funny_words_path=funny_words_path,
        preferences_path=preferences_path,
        human_examples_path=human_examples_path,
    )
```

### Pattern 2: Custom smolagents Model Adapter
**What:** Subclass `smolagents.Model`, implement `generate()`, use `openai.OpenAI` client internally.
**When to use:** Connecting to any OpenAI-compatible chat completions API.
**Confidence:** HIGH -- verified from smolagents v1.24.0 source code and official docs.

```python
# Source: smolagents v1.24.0 official docs + OpenAI Python client
import os
from openai import OpenAI
from smolagents import Model
from smolagents.models import ChatMessage, MessageRole

class OpenAICompatibleModel(Model):
    """Custom model adapter for any OpenAI-compatible chat completion API."""

    def __init__(
        self,
        model_name: str,
        api_base_url: str,
        api_key_env_var: str,
        **kwargs,
    ):
        super().__init__(model_id=model_name, **kwargs)
        api_key = os.environ.get(api_key_env_var)
        if not api_key:
            raise ValueError(f"Environment variable {api_key_env_var} not set")
        self.client = OpenAI(api_key=api_key, base_url=api_base_url)

    def generate(
        self,
        messages,
        stop_sequences=None,
        response_format=None,
        tools_to_call_from=None,
        **kwargs,
    ) -> ChatMessage:
        # Convert smolagents message format to OpenAI format
        openai_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                openai_messages.append(msg)
            elif hasattr(msg, "role") and hasattr(msg, "content"):
                openai_messages.append({"role": str(msg.role.value), "content": msg.content})
            else:
                openai_messages.append({"role": "user", "content": str(msg)})

        completion_kwargs = {
            "model": self.model_id,
            "messages": openai_messages,
            **self.kwargs,
            **kwargs,
        }
        if stop_sequences:
            completion_kwargs["stop"] = stop_sequences

        response = self.client.chat.completions.create(**completion_kwargs)
        content = response.choices[0].message.content

        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=content,
            tool_calls=None,
            raw=response,
        )


def create_model(config) -> OpenAICompatibleModel:
    """Factory function to create a model from AppConfig."""
    return OpenAICompatibleModel(
        model_name=config.model_name,
        api_base_url=config.api_base_url,
        api_key_env_var=config.api_key_env_var,
        max_tokens=4096,
        temperature=0.7,
    )
```

### Pattern 3: Settings File Layout
**What:** A single JSON settings file that points to all external config files.
**When to use:** CFG-04 requires all config files outside repo, referenced by path in a single settings file.

```json
{
    "funny_words_path": "/Users/patruff/chuckles-config/funny_words.json",
    "preferences_path": "/Users/patruff/chuckles-config/preferences.json",
    "human_examples_path": "/Users/patruff/chuckles-config/human_examples_clean.csv",
    "model_name": "qwen-3-32b",
    "api_base_url": "https://api.cerebras.ai/v1",
    "api_key_env_var": "CEREBRAS_API_KEY"
}
```

Default location: `~/.chuckles_prime/settings.json` (or passed via CLI flag `--settings`).

### Anti-Patterns to Avoid
- **Hardcoding funny words in source code:** The existing `word_structures.py` has `FUNNY_WORDS_BY_CATEGORY` hardcoded. Phase 1 must load these from external JSON instead.
- **Using `cerebras-cloud-sdk` directly:** Locks to one backend. Use `openai` client with `base_url` for any OpenAI-compatible API.
- **Returning custom `ModelResponse` from model adapter:** smolagents v1.24+ expects `ChatMessage`. The old `CerebrasModel.__call__()` returning `ModelResponse(content=...)` will break with current smolagents.
- **Skipping `Model` base class:** The `Model.__call__` delegates to `generate()`. Agents call `model(messages, ...)` which routes to `generate()`. Must subclass `Model` properly.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| OpenAI-compatible API client | Custom HTTP requests with `requests` | `openai` Python client | Handles auth, retries, streaming, error types, type hints. Every OpenAI-compatible API (Cerebras, Together, etc.) works with this client via `base_url`. |
| CSV parsing | Manual string splitting | Python stdlib `csv.DictReader` | Handles quoting, escaping, encodings correctly |
| Package building | `setup.py` | `pyproject.toml` with setuptools | Modern standard, declarative, no executable code in build |
| Message format conversion | Custom dict building | smolagents built-in `ChatMessage.from_dict()` | Handles tool calls, roles, nested content |
| Model base class plumbing | Custom `__call__` wrapper | Subclass `smolagents.Model` | Provides `__call__` -> `generate()` dispatch, `parse_tool_calls()`, `to_dict()`, `flatten_messages_as_text` support |

**Key insight:** The `openai` Python client is the universal adapter for any OpenAI-compatible API. Setting `base_url` is all that's needed to switch between Cerebras, Together, local vLLM, etc. Don't use provider-specific SDKs.

## Common Pitfalls

### Pitfall 1: Old smolagents Model Interface
**What goes wrong:** Using `__call__` with `ModelResponse` return type instead of `generate()` with `ChatMessage` return type. Agent crashes at first inference call.
**Why it happens:** The existing `CerebrasModel` was written for an older smolagents version. smolagents v1.24+ changed the interface.
**How to avoid:** Subclass `smolagents.Model`, implement `generate()`, return `ChatMessage(role=MessageRole.ASSISTANT, content=...)`.
**Warning signs:** `AttributeError: 'ModelResponse' object has no attribute 'role'` or `'content'` type mismatches.

### Pitfall 2: CSV Format Zones
**What goes wrong:** Treating the entire human_parodies.csv as a uniform 3-column CSV. Rows 14-24 have markdown-formatted entries. Rows 25-68 have data split across CSV columns incorrectly. Rows 69+ have triple-quoted strings with `Parody: "..."**` format.
**Why it happens:** The CSV was assembled from multiple LLM outputs with different formatting.
**How to avoid:** Parse in zones or use regex-based extraction that handles all formats. See "CSV Cleaning Strategy" section below.
**Warning signs:** Empty `output` fields, explanation text appearing in wrong columns, markdown artifacts like `13. **` prefixes.

### Pitfall 3: Encoding Issues in CSV
**What goes wrong:** Characters like `A(c)` appearing instead of proper accent marks (e.g., "risquA(c)" should be "risque").
**Why it happens:** UTF-8/Latin-1 encoding mismatch during original CSV creation.
**How to avoid:** Read with `encoding='utf-8'`, then clean known encoding artifacts with string replacement.
**Warning signs:** `A(c)` patterns in text, garbled accented characters.

### Pitfall 4: Config Paths Must Be Absolute or Resolved
**What goes wrong:** Relative paths in settings.json break when the working directory changes.
**Why it happens:** User runs the tool from different directories.
**How to avoid:** Resolve all paths relative to the settings file location, or require absolute paths. Use `Path.resolve()`.
**Warning signs:** `FileNotFoundError` when running from a different directory than expected.

### Pitfall 5: API Key Not in Environment
**What goes wrong:** `create_model()` fails because the API key env var is not set.
**Why it happens:** Settings file references `"api_key_env_var": "CEREBRAS_API_KEY"` but the variable is not exported.
**How to avoid:** Validate at load time with a clear error message. Check `os.environ.get(api_key_env_var)` in `create_model()` and raise `ValueError` with the env var name.
**Warning signs:** Generic `openai.AuthenticationError` or `None` API key passed to client.

## Code Examples

### Example 1: Minimal pyproject.toml
```toml
# Source: Python Packaging User Guide + smolagents pyproject.toml reference
[build-system]
requires = ["setuptools>=68.0", "setuptools-scm"]
build-backend = "setuptools.build_meta"

[project]
name = "chucklesPRIME"
version = "0.1.0"
description = "Phonetically-sound parody title generator with RLVR dataset output"
readme = "README.md"
requires-python = ">=3.10"
license = "MIT"
dependencies = [
    "smolagents>=1.24.0",
    "openai>=1.0.0",
    "rich",
    "pronouncing",
]

[project.optional-dependencies]
dev = ["pytest", "ruff"]

[project.scripts]
chuckles = "chuckles_prime.cli:main"

[tool.setuptools.packages.find]
where = ["src"]
```

### Example 2: CSV Cleaning Function
```python
# Regex-based extraction for all four CSV format zones
import re
import csv

def clean_human_parodies(input_path: str, output_path: str) -> int:
    """Clean human_parodies.csv into uniform (input, output, explanation) rows.

    Returns number of cleaned rows.
    """
    cleaned = []

    with open(input_path, encoding="utf-8") as f:
        raw_lines = f.readlines()

    for i, line in enumerate(raw_lines):
        if i == 0:  # Skip header
            continue

        line = line.strip()
        if not line:
            continue

        row = None

        # Zone 1 (rows 2-13): Clean CSV - input,output,explanation
        # Zone 4 (rows ~500+): Also clean CSV - Input,Output,Explanation
        # Try standard CSV parse first
        try:
            parsed = list(csv.reader([line]))[0]
        except:
            continue

        if len(parsed) >= 3:
            col0, col1, col2 = parsed[0].strip(), parsed[1].strip(), parsed[2].strip()

            # Zone 2 (rows 14-24): Numbered markdown like "13. **Chinchilla,Chintrilla**"
            md_match = re.match(r'^\d+\.\s*\*\*(.+?),\s*(.+?)\*\*$', col0)
            if md_match:
                row = (md_match.group(1).strip(), md_match.group(2).strip(), col1.strip('"'))
            # Zone 3 (rows 25-68): "Input / Output: Explanation: text" split across columns
            elif " / " in col0 and ":" in col0:
                slash_match = re.match(r'^(.+?)\s*/\s*(.+?):\s*(?:Detailed\s+)?Explanation:\s*(.+)', col0)
                if slash_match:
                    inp = slash_match.group(1).strip()
                    out = slash_match.group(2).strip()
                    expl = slash_match.group(3).strip()
                    # Reassemble explanation from remaining columns
                    full_expl = expl + " " + " ".join(c.strip() for c in parsed[1:] if c.strip())
                    row = (inp, out, clean_explanation(full_expl))
            # Zone 5 (rows 69+): Triple-quoted """input""" with Parody: ""output""**
            elif col0.startswith('"') and 'Parody:' in col1:
                inp = col0.strip('"').strip()
                parody_match = re.search(r'Parody:\s*"*(.+?)"*\*{0,2}$', col1.strip('"'))
                out = parody_match.group(1).strip() if parody_match else col1.strip('"').strip()
                expl = col2.strip('"').strip()
                row = (inp, out, expl)
            else:
                # Default: treat as standard 3-column CSV
                row = (col0, col1, col2)

        if row and row[0] and row[1]:  # Must have input and output
            inp, out, expl = row
            # Clean encoding artifacts
            expl = expl.replace("A(c)", "e")  # risquA(c) -> risque
            cleaned.append({"input": inp, "output": out, "explanation": expl})

    # Write cleaned CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["input", "output", "explanation"])
        writer.writeheader()
        writer.writerows(cleaned)

    return len(cleaned)


def clean_explanation(text: str) -> str:
    """Clean explanation text: fix encoding, remove artifacts."""
    text = text.replace("A(c)", "e")
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.rstrip(".")
    return text + "."
```

### Example 3: Creating the Model and Testing Connectivity
```python
# Source: openai Python client docs + smolagents v1.24 docs
import os
from chuckles_prime.config import load_config
from chuckles_prime.model import create_model

config = load_config("~/.chuckles_prime/settings.json")
model = create_model(config)

# Test with a simple prompt
response = model([{"role": "user", "content": "Say hello"}])
print(response.content)  # Should print model's response
```

## CSV Cleaning Strategy (Detailed)

### Format Zone Analysis

The human_parodies.csv (1,234 lines including header) has **five distinct zones**:

| Zone | Rows | Format | Example |
|------|------|--------|---------|
| Header | 1 | `input,output,explanation` | Column names |
| Zone 1 | 2-13 | Clean 3-column CSV | `Wolverine,Pullverine,"The term 'Wolverine' is..."` |
| Zone 2 | 14-24 | Numbered markdown in col1 | `"13. **Chinchilla,Chintrilla**","The parody..."` |
| Zone 3 | 25-68 | Slash-separated with explanation split across columns | `Chinchilla / Chintrilla: Detailed Explanation: The...,lters 'chinchilla'...,amusing juxtaposition...` |
| Zone 4 | 69-~270 | Triple-quoted with `Parody: "output"**` | `"""green peace""","Parody: ""green piss""**","This parody..."` |
| Zone 5 | ~270-1234 | Clean 3-column CSV (different sources) | `Pattern Recognition,Slattern Recognition,"The humor lies in..."` |

### Key Issues to Fix
1. **Zone 2 markdown artifacts:** Strip `\d+. **...**` formatting, extract input/output from markdown bold text
2. **Zone 3 split explanations:** Explanation text is broken across CSV columns due to commas in the text not being properly quoted
3. **Zone 4 triple quoting:** `"""input"""` and `Parody: ""output""**` formatting with markdown bold artifacts
4. **Encoding:** `A(c)` appears instead of `e` (acute accent) throughout -- simple string replacement
5. **Duplicates:** Some entries appear in multiple zones (e.g., "Spartacus/Fartacus" appears 3 times) -- deduplicate by (input, output) pair

### Cleaning Approach
1. **Read raw file** line by line (not via csv.reader for the whole file, since formats vary)
2. **Detect zone** per line based on format patterns
3. **Extract (input, output, explanation)** with zone-specific regex
4. **Normalize:** strip quotes, markdown artifacts, fix encoding
5. **Deduplicate** by (input.lower(), output.lower()) keeping the first occurrence
6. **Write** to clean CSV with proper quoting

### Expected Result
After cleaning: approximately 1,100-1,200 unique (input, output, explanation) rows in a clean 3-column CSV.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `smolagents.Model.__call__()` returning custom response | `smolagents.Model.generate()` returning `ChatMessage` | smolagents ~v1.10+ | Must rewrite model adapter |
| `cerebras-cloud-sdk` for Cerebras only | `openai` client with `base_url` for any backend | Always available | Enables backend flexibility |
| `setup.py` + `requirements.txt` | `pyproject.toml` (PEP 621) | 2022+ standard | Modern packaging |
| Hardcoded config in source | External JSON config with settings pointer | New for this project | Config isolation |

**Deprecated/outdated:**
- `ModelResponse` dataclass: Was used in old `CerebrasModel` -- replaced by `smolagents.ChatMessage`
- `setup.py`/`setup.cfg`: Still works but `pyproject.toml` is the modern standard
- `cerebras-cloud-sdk` dependency: Replaced by `openai` client with `base_url`

## Open Questions

1. **smolagents `tools_to_call_from` in generate()**
   - What we know: The `generate()` method receives `tools_to_call_from` parameter for tool-calling agents
   - What's unclear: For `CodeAgent` (which writes tool calls as Python code, not JSON), this parameter may not be used. Need to verify if our adapter needs to handle it.
   - Recommendation: Accept the parameter in `generate()` signature but don't pass it to the OpenAI client unless using `ToolCallingAgent`. CodeAgent extracts code from the text response directly.

2. **Message format conversion details**
   - What we know: smolagents passes `list[ChatMessage]` or `list[dict]` to `generate()`. The OpenAI client expects `list[dict]` with `role`/`content` keys.
   - What's unclear: How smolagents handles `MessageRole.TOOL_CALL` and `MessageRole.TOOL_RESPONSE` roles -- these are not standard OpenAI roles.
   - Recommendation: Map `TOOL_CALL` -> `assistant` and `TOOL_RESPONSE` -> `user` (or use `custom_role_conversions` if subclassing `ApiModel`). Verify with integration test.

3. **CSV cleaning accuracy**
   - What we know: Five format zones identified with specific patterns
   - What's unclear: Edge cases in zones 3 and 4 where explanation text contains commas/quotes that break CSV parsing
   - Recommendation: Write the cleaning script, run it, manually inspect a sample of 50 rows from each zone. Expect ~95% accuracy on first pass.

4. **Settings file default location**
   - What we know: Need a single settings file that points to external configs
   - What's unclear: Whether `~/.chuckles_prime/settings.json` or a CLI `--settings` flag is the right default
   - Recommendation: Support both. Default to `~/.chuckles_prime/settings.json`, allow override via `--settings` CLI flag or `CHUCKLES_SETTINGS` env var.

## Sources

### Primary (HIGH confidence)
- [smolagents v1.24.0 Models Reference](https://huggingface.co/docs/smolagents/en/reference/models) -- Model base class, generate() signature, ChatMessage return type, OpenAIModel implementation
- [smolagents v1.24.0 source code (models.py)](https://github.com/huggingface/smolagents/blob/v1.24.0/src/smolagents/models.py) -- Full Model class, ChatMessage dataclass, MessageRole enum, OpenAIModel.generate() implementation
- [Python Packaging User Guide: pyproject.toml](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/) -- pyproject.toml format, [project.scripts], build-system
- [Python docs: dataclasses](https://docs.python.org/3/library/dataclasses.html) -- Frozen dataclass, field(), default_factory
- [OpenAI Python client (GitHub)](https://github.com/openai/openai-python) -- OpenAI() constructor with api_key and base_url
- Direct inspection of `/Users/patruff/chucklesPRIME/parodies2026/generate_parody.py` -- Existing CerebrasModel adapter (old interface)
- Direct inspection of `/Users/patruff/chucklesPRIME/parodies2026/human_parodies/human_parodies.csv` -- All 1,234 rows examined for format patterns

### Secondary (MEDIUM confidence)
- [setuptools pyproject.toml docs](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html) -- Package discovery with src layout
- [smolagents pyproject.toml](https://github.com/huggingface/smolagents/blob/main/pyproject.toml) -- Reference for real-world pyproject.toml with CLI entry points

### Tertiary (LOW confidence)
- None -- all findings verified with primary sources.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- verified from official smolagents docs, openai client docs, Python packaging guide
- Architecture: HIGH -- patterns derived from smolagents source code and Python stdlib docs
- CSV cleaning: HIGH -- based on direct inspection of all 1,234 rows of the actual data file
- Pitfalls: HIGH -- identified from comparing old CerebrasModel code against current smolagents interface

**Research date:** 2026-01-31
**Valid until:** 2026-03-01 (smolagents is fast-moving; check for breaking changes in v1.25+)
