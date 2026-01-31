# Phase 2: Generation Engine - Research

**Researched:** 2026-01-31
**Domain:** smolagents CodeAgent orchestration, HF Hub tool loading, reasoning trace capture
**Confidence:** HIGH

## Summary

This research covers how to build the Generation Engine for chucklesPRIME using the smolagents v1.24.0 CodeAgent framework. The engine must read titles from CSV, generate 2 parody candidates per title using HF Hub phonetic tools, and capture full reasoning traces as structured data.

The standard approach uses `CodeAgent` with the `instructions` parameter (NOT the old `system_prompt` kwarg) to inject parody-specific guidance into smolagents' built-in ReAct loop. Tools are loaded from HuggingFace Hub via `load_tool()` with `trust_remote_code=True`. Full reasoning traces are captured via `return_full_result=True` which returns a `RunResult` object containing serialized `ActionStep` dicts with `model_output`, `code_action`, `observations`, and `tool_calls` fields. The existing `parodies2026/` code provides a proven prompt structure and generation flow that we adapt for the new architecture.

Key finding: The existing code uses an OLD smolagents API (`system_prompt` parameter, `step_callbacks` as list) that still works via kwargs but the canonical v1.24.0 approach uses `instructions` for custom prompt injection and `return_full_result=True` for trace capture. The `step_callbacks` API still works but is not needed for trace capture -- `RunResult.steps` provides everything.

**Primary recommendation:** Use `CodeAgent(tools=[...], model=model, instructions=custom_instructions, return_full_result=True)` and extract traces from `RunResult.steps`, which contains serialized `ActionStep` dicts with all model outputs, code actions, observations, and tool calls.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| smolagents | 1.24.0 | Agent framework (CodeAgent) | Already installed, Phase 1 model adapter built for it |
| openai | (installed) | LLM API client | Used by OpenAICompatibleModel from Phase 1 |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `patruff/parody-suggestions` (HF Hub) | latest | Phonetic parody word suggestion tool | Every generation -- pre-computes candidate replacements |
| `patruff/word-phone` (HF Hub) | latest | Phonetic similarity scoring tool | Every generation -- agent verifies replacements |

### HF Hub Tool Details (VERIFIED)

**`patruff/parody-suggestions`** loads as tool name `parody_word_suggester`:
- Inputs: `target` (str), `word_list_str` (JSON str), `min_similarity` (str, default "0.6"), `custom_phones` (dict, nullable)
- Output: JSON string with `target`, `target_phones`, `suggestions[]`
- Returns phonetically similar funny words for a target word

**`patruff/word-phone`** loads as tool name `word_phonetic_analyzer`:
- Inputs: `word` (str), `compare_to` (str, nullable), `custom_phones` (dict, nullable)
- Output: JSON string with phoneme analysis and comparison statistics
- Returns phonetic similarity score when `compare_to` is provided

**CRITICAL NOTE:** The existing parodies2026 code references these as `parody_tool` and `word_phone_tool`, but the actual tool names registered in smolagents are `parody_word_suggester` and `word_phonetic_analyzer`. The CodeAgent will use the tool names from the loaded tool objects.

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| CodeAgent | ToolCallingAgent | CodeAgent is better for multi-step reasoning with intermediate computations; ToolCallingAgent uses JSON tool calls which are simpler but less flexible |
| return_full_result | step_callbacks | step_callbacks add complexity; RunResult.steps provides the same data after completion |
| OpenTelemetry tracing | RunResult.steps | OpenTelemetry is overkill for batch processing; RunResult gives us everything inline |

**Installation:**
```bash
# Already installed via pyproject.toml
pip install smolagents openai
# setuptools needed for HF Hub tool loading (pkg_resources dependency)
pip install setuptools
```

## Architecture Patterns

### Recommended Project Structure
```
src/chuckles_prime/
    config.py            # Phase 1 (exists)
    model.py             # Phase 1 (exists)
    csv_cleaner.py       # Phase 1 (exists)
    prompts.py           # NEW: System prompt instructions + generation prompt builder
    tools.py             # NEW: HF Hub tool loading (thin wrapper)
    generator.py         # NEW: Core generation engine (CodeAgent orchestration)
    types.py             # NEW: GenerationRecord dataclass + supporting types
```

### Pattern 1: Instructions-Based Prompt Injection
**What:** Use the `instructions` parameter on CodeAgent to inject parody-specific guidance into the default smolagents ReAct system prompt. Do NOT replace the full system prompt.
**When to use:** Always -- this is the canonical v1.24.0 approach.
**Why:** The default CodeAgent system prompt contains critical ReAct loop instructions, code block formatting rules, and tool usage examples. Replacing it entirely (as the old code did with `system_prompt=`) risks breaking the agent's ability to parse its own output. The `instructions` parameter appends custom text at the end of the default system prompt via the `{{custom_instructions}}` template variable.

**Example:**
```python
# Source: smolagents v1.24.0 CodeAgent.initialize_system_prompt() (verified from source)
from smolagents import CodeAgent, load_tool
from chuckles_prime.model import create_model
from chuckles_prime.config import load_config

config = load_config("settings.json")
model = create_model(config)

# Load tools from HF Hub
parody_tool = load_tool("patruff/parody-suggestions", trust_remote_code=True)
phone_tool = load_tool("patruff/word-phone", trust_remote_code=True)

# Custom instructions injected into default system prompt
custom_instructions = """
You are a parody title generator. Your goal is to create phonetically similar
but humorous parody versions of movie/show titles.

WORKFLOW:
1. Use parody_word_suggester to get candidate funny words for each word in the title
2. Use word_phonetic_analyzer to verify phonetic similarity (score > 0.6 is acceptable)
3. Combine the best replacements into 2 final parody candidates
4. Return results as a JSON string with final_answer()

OUTPUT FORMAT (pass as string to final_answer):
{
    "parody1": "First Parody Title",
    "parody2": "Second Parody Title",
    "attempts": [
        {"text": "...", "scores": {"word1": 0.85}, "humor_note": "..."},
        ...
    ]
}
"""

agent = CodeAgent(
    tools=[parody_tool, phone_tool],
    model=model,
    instructions=custom_instructions,
    additional_authorized_imports=["json"],
    return_full_result=True,
    max_steps=15,
)

result = agent.run(f"Create funny parodies of the title: 'The Matrix'")
# result is RunResult with .output and .steps
```

### Pattern 2: RunResult Trace Extraction
**What:** Use `return_full_result=True` to get a `RunResult` object, then extract structured traces from `.steps`.
**When to use:** Every generation call -- this is how we capture GEN-03 (reasoning traces).

**Example:**
```python
# Source: smolagents v1.24.0 RunResult and ActionStep (verified from source inspection)
from smolagents import RunResult, ActionStep

# result = agent.run(task, return_full_result=True)  -- or set at init
# result.output -> final answer (the parody JSON string)
# result.steps -> list of step dicts (serialized ActionStep/TaskStep/PlanningStep)
# result.state -> "success" or "max_steps_error"
# result.token_usage -> TokenUsage(input_tokens=..., output_tokens=...)

# Each ActionStep dict contains:
# {
#   "step_number": int,
#   "model_output": str,          # LLM's raw thought + code text
#   "code_action": str,           # Extracted Python code
#   "observations": str,          # Print output from code execution
#   "tool_calls": [{"name": str, "arguments": Any, "id": str}],
#   "action_output": Any,         # Return value of code execution
#   "is_final_answer": bool,
#   "token_usage": {"input_tokens": int, "output_tokens": int},
#   "model_input_messages": [...], # Full message list sent to LLM
#   "error": {...} | null,
# }
```

### Pattern 3: GenerationRecord as Structured Output
**What:** Define a `GenerationRecord` dataclass that captures everything downstream phases need.
**When to use:** Every generation -- this is the interface between Phase 2 and Phase 3.

**Example:**
```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class ParodyCandidate:
    """A single parody candidate with its quality signals."""
    text: str
    phonetic_scores: dict[str, float]  # original_word -> similarity_score
    humor_note: str = ""

@dataclass
class AgentTrace:
    """Captured reasoning trace from one CodeAgent run."""
    steps: list[dict[str, Any]]       # Raw step dicts from RunResult
    final_output: str                  # Raw final_answer string
    token_usage: dict[str, int] | None # {input_tokens, output_tokens}
    state: str                         # "success" or "max_steps_error"

@dataclass
class GenerationRecord:
    """One generation result, ready for downstream dataset conversion."""
    input_title: str
    candidates: list[ParodyCandidate]  # Target: 2 candidates
    trace: AgentTrace
    model_name: str
    error: str | None = None           # Non-None if generation failed
```

### Pattern 4: Pre-computation of Suggestions Outside Agent
**What:** Call `parody_word_suggester` OUTSIDE the agent loop to pre-compute word suggestions, then pass them as context in the task prompt. The agent only uses `word_phonetic_analyzer` for verification.
**When to use:** This matches the existing parodies2026 pattern and is MORE EFFICIENT because: (a) it reduces agent steps (fewer tool calls = fewer LLM round-trips), (b) the suggestion tool needs the full funny_words list which is config data, not something the agent should manage.

**Example:**
```python
import json

def pre_compute_suggestions(title: str, funny_words: dict, parody_tool) -> dict:
    """Pre-compute parody suggestions for each word in title."""
    word_list_str = json.dumps(
        [w for words in funny_words.values() for w in words]
    )
    suggestions = {}
    for word in title.split():
        result = parody_tool.forward(
            target=word,
            word_list_str=word_list_str,
            min_similarity="0.5",
        )
        suggestions[word] = json.loads(result)
    return suggestions
```

### Anti-Patterns to Avoid
- **Replacing the full system prompt:** Do NOT set `prompt_templates={"system_prompt": custom_text, ...}` to override the entire system prompt. Use `instructions=` instead. The default system prompt has critical ReAct formatting instructions.
- **Using system_prompt= kwarg:** The old code uses `CodeAgent(system_prompt=...)`. In v1.24.0, there is no `system_prompt` parameter on CodeAgent. It may still work via **kwargs but is not the intended API.
- **Parsing LLM text output with regex:** The old code uses complex regex patterns to extract parodies from free-text LLM output. Instead, instruct the agent to use `final_answer()` with a structured JSON string and parse that.
- **Using step_callbacks for trace capture:** While `step_callbacks` still works, it adds complexity. `return_full_result=True` gives you the same data in a cleaner way.
- **Loading tools inside the agent code:** Tools should be loaded once at initialization, not inside each agent.run() call. The agent's code execution sandbox does NOT have access to `load_tool`.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| LLM message formatting | Custom message conversion | `Model._prepare_completion_kwargs()` | Already handles role mapping, stop sequences, tool descriptions (Phase 1) |
| Agent ReAct loop | Custom think-act-observe loop | `CodeAgent.run()` | smolagents handles multi-step reasoning, error recovery, max_steps limits |
| Tool call parsing | Regex extraction of tool calls from text | `RunResult.steps[n]["tool_calls"]` | smolagents already parses and records tool calls |
| Phonetic similarity | Custom phoneme comparison | `patruff/word-phone` HF Hub tool | CMU dictionary lookup, weighted phone comparison, already deployed |
| Word suggestion | Custom word matching | `patruff/parody-suggestions` HF Hub tool | Phonetic matching with similarity thresholds, already deployed |
| System prompt templating | String formatting with `{template}` | smolagents Jinja2 templates + `instructions` param | Auto-injects tool descriptions, import lists, code block tags |
| Output capture | stdout redirect / regex on logs | `RunResult.steps` from `return_full_result=True` | Structured dicts with all fields, serializable to JSON |

**Key insight:** The existing parodies2026 code hand-rolls output capture (OutputCapture class with regex), prompt formatting (string templates with `{{authorized_imports}}`), and result extraction (regex on free text). ALL of these have cleaner solutions in smolagents v1.24.0.

## Common Pitfalls

### Pitfall 1: Old API vs New API Confusion
**What goes wrong:** Copying patterns from parodies2026 that use deprecated or changed smolagents API (e.g., `system_prompt=`, `step_callbacks` as primary trace mechanism, old ModelResponse return type).
**Why it happens:** The existing code was written for an older smolagents version with a different API surface.
**How to avoid:** Use ONLY: `instructions=` for custom prompts, `return_full_result=True` for traces, `ChatMessage` for model responses (already done in Phase 1).
**Warning signs:** TypeError on CodeAgent init, missing attributes on result objects.

### Pitfall 2: Tool Name Mismatch
**What goes wrong:** Code references tools as `word_phone_tool(...)` or `parody_tool(...)` but the actual smolagents tool names (used in the agent's code sandbox) are `word_phonetic_analyzer(...)` and `parody_word_suggester(...)`.
**Why it happens:** The old code uses variable names that don't match the tool's registered `name` attribute. In CodeAgent, tools are called by their `.name` property, not the Python variable name.
**How to avoid:** Always check `tool.name` after loading. Reference tools by their registered name in prompts/instructions. The CodeAgent system prompt auto-generates tool signatures from the loaded tool objects.
**Warning signs:** Agent writes code calling `word_phone_tool(...)` but gets NameError because the tool is registered as `word_phonetic_analyzer`.

### Pitfall 3: Agent Writes Non-Parseable Output
**What goes wrong:** The agent returns free-text parody descriptions instead of structured JSON via `final_answer()`.
**Why it happens:** Insufficient instructions about output format; LLM decides to be "creative" with output.
**How to avoid:** Be explicit in `instructions` about the exact JSON format expected. Include an example. Tell the agent to call `final_answer(json.dumps({...}))`.
**Warning signs:** `result.output` is a narrative string instead of parseable JSON.

### Pitfall 4: pkg_resources ImportError on Tool Loading
**What goes wrong:** `load_tool("patruff/...", trust_remote_code=True)` fails with `No module named 'pkg_resources'`.
**Why it happens:** The HF Hub tools depend on `pkg_resources` which is in `setuptools`, not always installed in modern Python environments.
**How to avoid:** Ensure `setuptools` is in dependencies. Add it to pyproject.toml if not already there.
**Warning signs:** ImportError on first tool load.

### Pitfall 5: Agent Runs Out of Steps
**What goes wrong:** Agent hits `max_steps` (default 20) without producing a final answer, resulting in `state="max_steps_error"`.
**Why it happens:** Agent goes on tangents, retries failed tool calls excessively, or gets stuck in reasoning loops.
**How to avoid:** Set `max_steps=15` (reasonable for 2-candidate generation). Handle `max_steps_error` state gracefully -- record it in GenerationRecord.error. Make instructions concise and directive.
**Warning signs:** Many generations timing out, high token usage per generation.

### Pitfall 6: Forgetting trust_remote_code=True
**What goes wrong:** `load_tool("patruff/...", trust_remote_code=False)` silently fails or raises an error.
**Why it happens:** Security default in smolagents requires explicit opt-in for remote code.
**How to avoid:** Always pass `trust_remote_code=True` when loading from HF Hub. This is expected for tools you control.
**Warning signs:** Tool loading fails with permissions error.

### Pitfall 7: Batch Processing Without Error Isolation
**What goes wrong:** One failed generation crashes the entire batch run.
**Why it happens:** No try/except around individual title processing.
**How to avoid:** Wrap each generation in try/except, record errors in GenerationRecord.error, continue to next title. Log errors but don't stop the batch.
**Warning signs:** Partial output files, no results for titles after the first failure.

## Code Examples

Verified patterns from official sources and installed code inspection:

### Loading Tools from HF Hub
```python
# Source: smolagents v1.24.0 load_tool (verified from source inspection)
from smolagents import load_tool

parody_tool = load_tool("patruff/parody-suggestions", trust_remote_code=True)
phone_tool = load_tool("patruff/word-phone", trust_remote_code=True)

# IMPORTANT: Check actual tool names
print(parody_tool.name)  # "parody_word_suggester"
print(phone_tool.name)   # "word_phonetic_analyzer"
```

### Creating CodeAgent with Instructions
```python
# Source: smolagents v1.24.0 (verified from CodeAgent.__init__ and
# MultiStepAgent.__init__ source inspection)
from smolagents import CodeAgent

agent = CodeAgent(
    tools=[parody_tool, phone_tool],
    model=model,                              # Phase 1 OpenAICompatibleModel
    instructions=PARODY_INSTRUCTIONS,         # Custom instructions (str)
    additional_authorized_imports=["json"],    # Allow json in code sandbox
    return_full_result=True,                  # Get RunResult instead of just output
    max_steps=15,                             # Reasonable limit for parody generation
)
```

### Running Agent and Extracting Traces
```python
# Source: smolagents v1.24.0 RunResult (verified from source inspection)
result = agent.run(
    task=generation_prompt,
    return_full_result=True,  # Can also be set at init
)

# result is RunResult
assert result.state in ("success", "max_steps_error")

# Extract structured trace
trace = AgentTrace(
    steps=result.steps,                     # list[dict] -- serialized steps
    final_output=str(result.output),
    token_usage=result.token_usage.dict() if result.token_usage else None,
    state=result.state,
)

# Each step dict (for ActionStep) contains:
for step in result.steps:
    if "model_output" in step:  # ActionStep (not TaskStep)
        print(step["model_output"])    # LLM thought + code
        print(step["code_action"])     # Extracted code
        print(step["observations"])    # print() output from code
        print(step["tool_calls"])      # [{"name": ..., "arguments": ..., "id": ...}]
```

### Reading Input Titles from CSV
```python
# Standard Python CSV reading -- matches GEN-01 requirement
import csv
from pathlib import Path

def read_input_titles(csv_path: str | Path) -> list[str]:
    """Read titles from a CSV file with a 'title' column."""
    titles = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row.get("title", "").strip()
            if title:
                titles.append(title)
    return titles
```

### Batch Processing with Error Isolation
```python
# Pattern from existing batch_generate.py, adapted for new architecture
def generate_batch(
    titles: list[str],
    agent: CodeAgent,
    prompt_builder,  # callable(title) -> str
) -> list[GenerationRecord]:
    records = []
    for i, title in enumerate(titles):
        try:
            prompt = prompt_builder(title)
            result = agent.run(task=prompt, reset=True)
            record = parse_result(title, result)
        except Exception as e:
            record = GenerationRecord(
                input_title=title,
                candidates=[],
                trace=AgentTrace(steps=[], final_output="", token_usage=None, state="error"),
                model_name=agent.model.model_id,
                error=str(e),
            )
        records.append(record)
    return records
```

### Pre-computing Suggestions (Outside Agent)
```python
# Source: Adapted from parodies2026/generate_parody.py pattern (verified working)
import json

def build_generation_prompt(
    title: str,
    suggestions: dict,
    examples: list[tuple[str, str, str]],
    preferences_text: str,
) -> str:
    """Build the task prompt for the CodeAgent."""
    examples_text = "\n".join(
        f'  - "{orig}" -> "{parody}" ({reason})'
        for orig, parody, reason in examples[:10]
    )

    return f"""Create 2 funny parodies of the title: "{title}"

KNOWN GOOD EXAMPLES (learn from their style):
{examples_text}

STYLE PREFERENCES:
{preferences_text}

PRE-COMPUTED SUGGESTIONS (phonetically similar funny words for each word):
{json.dumps(suggestions, indent=2)}

INSTRUCTIONS:
1. Review the pre-computed suggestions above
2. For each promising replacement, verify with word_phonetic_analyzer(word=original, compare_to=replacement)
3. Score > 0.6 means acceptable phonetic similarity
4. Create at least 3 attempts, pick the 2 funniest
5. Call final_answer() with a JSON string containing your 2 best parodies

OUTPUT FORMAT (pass as JSON string to final_answer):
{{
    "parody1": "First Parody Title",
    "parody2": "Second Parody Title",
    "attempts": [
        {{"text": "Attempt Text", "scores": {{"word": 0.85}}, "humor_note": "Why it is funny"}}
    ]
}}
"""
```

## State of the Art

| Old Approach (parodies2026) | Current Approach (v1.24.0) | When Changed | Impact |
|---|---|---|---|
| `CodeAgent(system_prompt=...)` | `CodeAgent(instructions=...)` | smolagents ~v1.10+ | Custom instructions injected properly into default ReAct prompt |
| `step_callbacks=[callback]` for trace | `return_full_result=True` -> RunResult | smolagents v1.22+ | Cleaner trace access, RunResult has .steps with full dicts |
| `ModelResponse(content=str)` | `ChatMessage(role, content, tool_calls, raw)` | smolagents v1.x | Phase 1 already uses ChatMessage (correct) |
| OutputCapture regex parsing | `RunResult.steps[n]["model_output"]` etc. | smolagents v1.22+ | No regex needed, structured step data |
| `result.messages` | `result.steps` (messages deprecated 1.22, removed 1.25) | v1.22 | Use .steps not .messages |

**Deprecated/outdated:**
- `system_prompt` kwarg on CodeAgent: Use `instructions` instead
- `step_callbacks` as primary trace mechanism: Use `return_full_result=True`
- `RunResult.messages`: Deprecated in v1.22, use `RunResult.steps`
- Free-text output parsing with regex: Use structured JSON via `final_answer()`

## Open Questions

Things that could not be fully resolved:

1. **Exact output format from the agent**
   - What we know: We can instruct the agent to return JSON via `final_answer(json.dumps({...}))`. The `result.output` will be whatever the agent passes to `final_answer()`.
   - What's unclear: Whether the LLM (via Cerebras/other providers) will reliably produce valid JSON in the `final_answer` call. May need fallback parsing.
   - Recommendation: Always try `json.loads(result.output)` first; fall back to regex extraction on parse failure. Record parse failures in GenerationRecord.error.

2. **Token limits and generation costs**
   - What we know: Pre-computed suggestions + examples + instructions can be 2000+ tokens in the prompt. Each agent step uses LLM calls.
   - What's unclear: How many steps a typical generation takes and what the cost per title is.
   - Recommendation: Start with `max_steps=15`, monitor `result.token_usage`. Adjust based on empirical data.

3. **How to get 2 candidates reliably**
   - What we know: The prompt instructs the agent to produce 2 parodies. The existing code generates multiple attempts and picks best.
   - What's unclear: Whether agents reliably produce exactly 2 candidates vs 1 or 3.
   - Recommendation: Validate output and handle edge cases: if 1 candidate, record it; if 0, mark as error; if >2, take first 2.

4. **custom_phones dict passing to tools**
   - What we know: Both HF tools accept `custom_phones` as an optional dict parameter. The existing code passes this from `word_structures.py`.
   - What's unclear: Whether the CodeAgent can pass complex dict arguments to tools reliably in its generated code.
   - Recommendation: Pre-compute suggestions outside the agent (passing custom_phones directly), and only use `word_phonetic_analyzer` inside the agent (which may not need custom_phones for basic comparison).

## Sources

### Primary (HIGH confidence)
- smolagents v1.24.0 installed package -- source inspection of `CodeAgent.__init__`, `MultiStepAgent.__init__`, `RunResult`, `ActionStep`, `load_tool`, `initialize_system_prompt`
- smolagents v1.24.0 default prompts -- `code_agent.yaml` template with `{{custom_instructions}}` variable
- HF Hub tools verified -- `patruff/parody-suggestions` (name: `parody_word_suggester`), `patruff/word-phone` (name: `word_phonetic_analyzer`) both load and execute successfully
- Existing codebase -- `parodies2026/generate_parody.py`, `parodies2026/system_prompt.py`, `parodies2026/batch_generate.py`, `parodies2026/word_structures.py`, `parodies2026/test_popular_movies.py`
- Phase 1 code -- `src/chuckles_prime/config.py`, `src/chuckles_prime/model.py`, `src/chuckles_prime/csv_cleaner.py`

### Secondary (MEDIUM confidence)
- [HuggingFace smolagents API docs (v1.24.0)](https://huggingface.co/docs/smolagents/en/reference/agents) -- RunResult, step_callbacks, return_full_result parameter documentation
- [GitHub issue #322](https://github.com/huggingface/smolagents/issues/322) -- Discussion on capturing full reasoning traces
- [HF inspect_runs tutorial](https://huggingface.co/docs/smolagents/tutorials/inspect_runs) -- Confirms OpenTelemetry approach exists but is not needed for our batch use case

### Tertiary (LOW confidence)
- WebSearch results for smolagents trace capture patterns -- pointed to OpenTelemetry integrations (Langfuse, Phoenix, MLflow) which are overkill for our use case but confirm the trace capture challenge is real

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- smolagents v1.24.0 verified from installed package, HF Hub tools verified working
- Architecture: HIGH -- patterns derived from existing working code + verified v1.24.0 API inspection
- Pitfalls: HIGH -- several pitfalls identified from actual API differences between old code and current version
- Code examples: HIGH -- all examples verified against actual installed source code

**Research date:** 2026-01-31
**Valid until:** 2026-03-01 (smolagents moves fast, check for breaking changes in v1.25+)
