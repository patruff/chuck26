# Coding Conventions

**Analysis Date:** 2026-01-31

## Naming Patterns

**Files:**
- Lowercase with underscores: `generate_parody.py`, `word_structures.py`, `batch_generate.py`
- Tool classes use descriptive names: `WordPhoneTool`, `ParodyWordSuggestionTool`, `CerebrasModel`
- Test files follow naming pattern: `test_popular_movies.py` (executable tests, not unit test framework)

**Functions:**
- Snake_case: `generate_parody()`, `extract_parody_from_result()`, `load_known_parodies()`, `process_batch()`
- Prefix verbs for clarity: `get_*` for retrieval, `extract_*` for parsing, `process_*` for operations
- Private methods prefix with underscore: `_preprocess_content()`, `_get_word_phones()`, `_strip_stress()`
- Tool methods use `forward()` for main execution (required by smolagents Tool base class)

**Variables:**
- Snake_case: `api_key`, `output_dir`, `final_parody`, `raw_output_file`
- Constants in UPPERCASE with underscores: `AGENT_SYSTEM_PROMPT`, `PARODY_STYLE_GUIDE`, `VOWEL_REF`, `DEFAULT_TEMPLATE`
- Private module variables: `_vowels_match()`, class attributes in lowercase
- Loop variables: explicit names over single letters: `for word in words:` not `for w in words:`

**Types:**
- Dataclasses for structured data: `@dataclass RLVRDataPoint`, `@dataclass ToolCall`, `@dataclass ParodyAttempt`
- Type hints used throughout: `def generate_parody(title: str, model_name="qwen-3-32b", api_key=None, output_dir='./output') -> str:`
- Optional types imported: `from typing import List, Optional, Dict, Any, Tuple`
- Union types with pipe: `def get_parody_for_title(title: str) -> str | None:` (Python 3.10+ syntax)

## Code Style

**Formatting:**
- No automated formatter configured (no `.prettierrc`, `pylintrc`, or `black` config)
- Code follows PEP 8 conventions by convention (not enforcement)
- Line length appears unconstrained (some lines exceed 100 characters)
- Indentation: 4 spaces consistently

**Linting:**
- No linting configuration present (no `.eslintrc`, `pylint.ini`, `flake8` config)
- Code quality relies on manual review and convention adherence

## Import Organization

**Order:**
1. Standard library imports: `import os`, `import sys`, `import csv`, `import json`, `import re`, `import argparse`
2. Standard library typing: `from typing import List, Optional, Dict, Any, Tuple`
3. Third-party packages: `from pathlib import Path`, `from datetime import datetime`, `from dataclasses import dataclass, field`
4. Domain imports: `from smolagents import CodeAgent, load_tool, Tool`
5. External services: `from cerebras.cloud.sdk import Cerebras`
6. Local imports: `from word_structures import custom_phones`, `from system_prompt import AGENT_SYSTEM_PROMPT`

**Path Aliases:**
- `Path` from `pathlib` used for file operations: `MODULE_DIR = Path(__file__).parent`
- Relative module imports: `from generate_parody import generate_parody`

## Error Handling

**Patterns:**
- Broad exception catching: `except Exception as e:` (not specific exception types)
- Logging errors with `logging.error()` and traceback info: `logging.error(error_msg, exc_info=True)`
- Error messages included in return values: `return ModelResponse(content=f"Error: {str(e)}")`
- CSV output records errors as data: `{'input': title, 'parody_result': f"ERROR: {str(e)}", 'reasoning': "Generation failed due to error"}`
- Exit codes used in CLI: `sys.exit(1)` for environment variable failures
- Data point labeling for errors: `quality_label="error"` in RLVRDataPoint

**No exceptions raised** - errors are caught and converted to return values or logged

Example from `generate_parody.py`:
```python
except Exception as e:
    logging.error(f"Error calling Cerebras API: {str(e)}", exc_info=True)
    return ModelResponse(content=f"Error: {str(e)}")
```

## Logging

**Framework:** Standard `logging` module

**Configuration pattern** (all files use this):
```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)
```

**Patterns:**
- Log initialization: `logging.info(f"Initialized {ClassName}")`
- Log processing steps: `logging.info(f"[{i}/{len(items)}] Processing: {item}")`
- Log errors with context: `logging.error(f"Error processing '{title}': {e}")`
- Progress indicators with visual separators: `logging.info(f"\n{'='*80}")`
- Summary statistics logged at end: `logging.info(f"Saved {count} results")`

**Console output mixed with logging** - Rich library used for formatted console output: `from rich import print as rprint`

## Comments

**When to Comment:**
- Module-level docstrings explain purpose (seen in all files)
- Classes and functions documented with docstrings using triple quotes
- Section comments with visual separators: `# =============================================================================`
- Inline comments for complex regex patterns and calculations
- Docstrings include parameter types and return types

**JSDoc/TSDoc:**
- Not applicable to Python codebase
- Docstrings follow Python convention with triple quotes
- Example docstring format:
```python
def load_known_parodies() -> List[Tuple[str, str, str]]:
    """
    Load known funny parodies from CSV file.

    Returns:
        List of tuples: (original_title, parody_title, reasoning)
    """
```

## Function Design

**Size:**
- Functions range from 10 to 150+ lines
- Single responsibility principle observed: extraction functions, processing functions, output functions separate
- Larger functions (100+ lines) handle complex multi-step workflows (e.g., `process_batch()`, `run_popular_movies_test()`)

**Parameters:**
- Use keyword arguments with defaults: `def generate_parody(title: str, model_name="qwen-3-32b", api_key=None, output_dir='./output')`
- Dictionary parameters for complex configurations
- Type hints on all parameters
- Optional parameters defaulted to None then checked: `if not api_key:`

**Return Values:**
- Single return type per function
- Complex returns use dataclasses or dictionaries
- Error returns embedded in same type as success returns
- No exceptions raised - errors returned as values

## Module Design

**Exports:**
- No explicit `__all__` definitions
- Classes and functions used as defined (no limiting exports)
- Tool classes inherit from `smolagents.Tool` base class

**Barrel Files:**
- Not used - direct imports from individual modules
- Example: `from word_structures import custom_phones, funny_words, KNOWN_FUNNY_PARODIES`

## Special Patterns Observed

**Dataclass Usage:**
- Extensive use of `@dataclass` for data structures: `RLVRDataPoint`, `ToolCall`, `ParodyAttempt`, `RLVRTemplateTags`, `QualityCriteria`
- Dataclasses include type hints on all fields
- `field(default_factory=...)` used for mutable defaults

**Configuration as Code:**
- Constants defined at module top: `POPULAR_MOVIES = [...]`, `TEMPLATES = {...}`
- Template tags configurable via dataclass: `RLVRTemplateTags(reasoning_start="<start_working_out>", ...)`
- Multiple preset templates: `DEFAULT_TEMPLATE`, `DEEPSEEK_TEMPLATE`

**Regex Patterns:**
- Heavy use of regex for extraction: `r'### Final Chosen Parody:.*?\n\*\*"?([^"*\n]+)"?\*\*'`
- Patterns stored as module constants when used multiple times
- Flags used: `re.DOTALL`, `re.IGNORECASE` with `re.finditer()`, `re.search()`, `re.findall()`

**String Formatting:**
- f-strings exclusively: `f"Processing {i}/{len(titles)}: {title}"`
- Multiline strings for prompts: wrapped in triple quotes
- `.format()` method used in template strings: `GENERATION_PROMPT_TEMPLATE.format(title=title, examples_text=...)`

---

*Convention analysis: 2026-01-31*
