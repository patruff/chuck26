# Phase 4: Pipeline CLI - Research

**Researched:** 2026-01-31
**Domain:** Python CLI, pipeline orchestration, rich terminal output
**Confidence:** HIGH

## Summary

Phase 4 wires together the existing modules (config, model, tools, generator, traces, dataset) into a CLI with two subcommands: `generate` (full pipeline) and `convert` (dataset-only from saved JSONL). The project already declares `chuckles = "chuckles_prime.cli:main"` in pyproject.toml and already has `rich` as a dependency.

The recommended approach is **argparse** (stdlib) for command parsing -- it supports subcommands natively, adds zero new dependencies, and the CLI surface is small (2 subcommands, ~5 flags each). Rich handles all progress and summary output. JSONL deserialization uses a simple hand-rolled `_record_from_dict()` function since the dataclass nesting is only 2 levels deep and all types are primitives/lists/dicts.

**Primary recommendation:** Use argparse for CLI structure, rich.progress.Progress for per-title tracking, rich.table.Table for completion summaries, and a thin `_record_from_dict()` for JSONL deserialization. No new dependencies needed.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| argparse | stdlib | CLI argument parsing with subcommands | Zero deps, sufficient for 2 subcommands, already familiar pattern in codebase |
| rich | already installed | Progress bars, summary tables, colored console output | Already a project dependency, provides track()/Progress/Table |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pathlib | stdlib | Path resolution for settings and output files | All file path handling |
| sys | stdlib | sys.exit() for error codes | CLI exit on fatal errors |
| json | stdlib | JSONL reading for convert command | Deserializing archived traces |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| argparse | typer | Cleaner syntax but adds a dependency; overkill for 2 subcommands |
| argparse | click | More powerful but adds a dependency; unnecessary complexity here |
| hand-rolled deserializer | dataclasses-json / cattrs | Adds dependency for a problem that is 20 lines of code here |

**Installation:**
```bash
# No new dependencies needed -- argparse is stdlib, rich is already installed
```

## Architecture Patterns

### Recommended Project Structure
```
src/chuckles_prime/
    cli.py           # NEW: argparse setup, subcommand dispatch, progress output
    # All existing modules unchanged:
    config.py        # load_config()
    model.py         # create_model(), check_model_connectivity()
    generator.py     # read_input_titles(), create_agent(), generate_batch()
    tools.py         # load_parody_tools()
    traces.py        # archive_traces()
    dataset.py       # records_to_grpo_dataset(), build_dpo_dataset(), push_dataset()
    rewards.py       # compute_* functions (called by dataset.py internally)
    types.py         # ParodyCandidate, AgentTrace, GenerationRecord
    prompts.py       # PARODY_INSTRUCTIONS, build_generation_prompt()
```

### Pattern 1: Thin CLI Dispatching to Pipeline Functions
**What:** cli.py contains only argument parsing, progress display, and error handling. All business logic stays in existing modules.
**When to use:** Always -- the CLI is a thin shell over the existing pipeline.
**Example:**
```python
import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn
from rich.table import Table

console = Console()


def cmd_generate(args: argparse.Namespace) -> None:
    """Full pipeline: CSV -> generate -> archive -> dataset -> push."""
    from chuckles_prime.config import load_config
    from chuckles_prime.model import create_model
    from chuckles_prime.tools import load_parody_tools
    from chuckles_prime.generator import read_input_titles, create_agent, generate_single
    from chuckles_prime.traces import archive_traces
    from chuckles_prime.dataset import records_to_grpo_dataset, build_dpo_dataset, push_dataset

    config = load_config(args.settings)
    titles = read_input_titles(args.input)
    console.print(f"Loaded [bold]{len(titles)}[/bold] titles from {args.input}")

    model = create_model(config)
    parody_tool, phone_tool = load_parody_tools()
    agent = create_agent(model, parody_tool, phone_tool)

    # Generate with progress
    records = []
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generating", total=len(titles))
        for title in titles:
            record = generate_single(title, agent, parody_tool, config)
            records.append(record)
            status = "[green]OK[/green]" if not record.error else f"[red]{record.error}[/red]"
            progress.console.print(f"  {title} ... {status}")
            progress.advance(task)

    # Archive traces
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    traces_path = output_dir / "traces.jsonl"
    count = archive_traces(records, traces_path)
    console.print(f"Archived [bold]{count}[/bold] traces to {traces_path}")

    # Convert and push datasets
    # ... (dataset conversion and push logic)

    _print_summary(records)


def cmd_convert(args: argparse.Namespace) -> None:
    """Dataset-only: JSONL -> dataset -> push."""
    records = _load_records(Path(args.traces))
    # ... convert and push


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="chuckles",
        description="Generate phonetically sound parody titles and RLVR datasets",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # generate subcommand
    gen = sub.add_parser("generate", help="Full pipeline: CSV -> parodies -> datasets -> Hub")
    gen.add_argument("input", help="Path to input CSV with 'title' column")
    gen.add_argument("--settings", default="settings.json", help="Path to settings JSON (default: settings.json)")
    gen.add_argument("--output-dir", default="output", help="Directory for traces JSONL (default: output/)")
    gen.add_argument("--grpo-repo", default=None, help="HF Hub repo ID for GRPO dataset")
    gen.add_argument("--dpo-repo", default=None, help="HF Hub repo ID for DPO dataset")
    gen.add_argument("--no-push", action="store_true", help="Skip pushing to Hub")
    gen.set_defaults(func=cmd_generate)

    # convert subcommand
    conv = sub.add_parser("convert", help="Convert existing traces to datasets")
    conv.add_argument("traces", help="Path to traces JSONL file")
    conv.add_argument("--settings", default="settings.json", help="Path to settings JSON (for human examples)")
    conv.add_argument("--grpo-repo", default=None, help="HF Hub repo ID for GRPO dataset")
    conv.add_argument("--dpo-repo", default=None, help="HF Hub repo ID for DPO dataset")
    conv.add_argument("--no-push", action="store_true", help="Skip pushing to Hub")
    conv.set_defaults(func=cmd_convert)

    args = parser.parse_args()
    args.func(args)
```

### Pattern 2: Lazy Imports for Fast CLI Startup
**What:** Import heavy modules (smolagents, datasets, openai) inside command functions, not at module top level.
**When to use:** Always in cli.py -- `chuckles --help` should be instant, not wait for torch/smolagents to load.
**Example:**
```python
def cmd_generate(args):
    # Heavy imports only when actually running the command
    from chuckles_prime.config import load_config
    from chuckles_prime.model import create_model
    # ...
```

### Pattern 3: Rich Progress Replacing Print Statements
**What:** Use rich Progress context manager instead of the bare `print()` in `generate_batch()`. The CLI calls `generate_single()` directly in a loop with Progress tracking, bypassing `generate_batch()`.
**When to use:** In the CLI `cmd_generate` function. The existing `generate_batch()` with its `print()` calls remains available for non-CLI use.
**Example:**
```python
records = []
with Progress(
    TextColumn("[bold blue]{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    console=console,
) as progress:
    task = progress.add_task("Generating parodies", total=len(titles))
    for title in titles:
        try:
            record = generate_single(title, agent, parody_tool, config)
        except Exception as e:
            record = GenerationRecord(
                input_title=title,
                candidates=[],
                trace=AgentTrace(steps=[], final_output="", token_usage=None, state="error"),
                model_name=config.model_name,
                error=str(e),
            )
        status = "[green]OK" if not record.error else f"[red]ERR: {record.error}"
        progress.console.print(f"  {title} ... {status}")
        progress.advance(task)
        records.append(record)
```

### Pattern 4: Summary Table on Completion
**What:** Print a rich.table.Table showing aggregate statistics after generation completes.
**When to use:** At the end of both `generate` and `convert` commands.
**Example:**
```python
def _print_summary(records: list) -> None:
    successes = sum(1 for r in records if r.error is None)
    failures = len(records) - successes
    total_candidates = sum(len(r.candidates) for r in records)

    table = Table(title="Generation Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Titles processed", str(len(records)))
    table.add_row("Successful", f"[green]{successes}[/green]")
    table.add_row("Failed", f"[red]{failures}[/red]" if failures else "0")
    table.add_row("Total candidates", str(total_candidates))
    console.print(table)
```

### Anti-Patterns to Avoid
- **Modifying `generate_batch()` to accept a progress callback:** Adds coupling. Instead, call `generate_single()` in a loop from the CLI with its own error isolation. Keep `generate_batch()` unchanged for non-CLI usage.
- **Top-level imports of heavy modules in cli.py:** Makes `chuckles --help` slow. Use lazy imports inside command functions.
- **Putting business logic in cli.py:** The CLI should only parse args, call existing functions, and display output. No generation/dataset/reward logic in cli.py.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| CLI argument parsing | Custom sys.argv parser | argparse (stdlib) | Handles help text, validation, subcommands, type conversion |
| Progress display | print() with counters | rich.progress.Progress | Flicker-free, time estimates, bar rendering |
| Summary output | print() with f-strings | rich.table.Table | Aligned columns, borders, colored cells |
| Console styling | ANSI escape codes | rich.console.Console | Cross-platform, auto-detects terminal capabilities |

**Exception -- DO hand-roll:** JSONL deserialization of GenerationRecord. The nesting is only 2 levels (GenerationRecord -> ParodyCandidate/AgentTrace), all fields are primitives/lists/dicts, and adding `dataclasses-json` or `cattrs` for this is not worth the dependency. A 20-line `_record_from_dict()` function is sufficient and more transparent.

## Common Pitfalls

### Pitfall 1: Slow CLI Startup from Heavy Imports
**What goes wrong:** Importing smolagents/openai/datasets at module level makes even `chuckles --help` take 3-5 seconds.
**Why it happens:** These libraries load torch, tokenizers, etc. on import.
**How to avoid:** Use lazy imports inside command handler functions, not at cli.py top level.
**Warning signs:** `chuckles --help` taking more than 0.5 seconds.

### Pitfall 2: Progress Bar Conflict with Agent Print Output
**What goes wrong:** smolagents CodeAgent may print debug info to stdout, which interleaves with rich Progress display and corrupts the terminal output.
**Why it happens:** Rich Progress uses terminal control sequences that assume exclusive stdout control.
**How to avoid:** Use `progress.console.print()` for any output during progress tracking. Consider redirecting agent verbose output or setting agent verbosity to minimum. The `console=console` parameter on Progress ensures all output goes through the same Console instance.
**Warning signs:** Garbled terminal output during generation.

### Pitfall 3: JSONL Deserialization Losing Nested Types
**What goes wrong:** `json.loads()` returns plain dicts, not dataclass instances. Code that accesses `record.candidates[0].text` fails with `AttributeError: 'dict' has no attribute 'text'`.
**Why it happens:** `dataclasses.asdict()` + `json.dumps()` loses type information on serialization; `json.loads()` has no way to reconstruct it.
**How to avoid:** Write an explicit `_record_from_dict(d: dict) -> GenerationRecord` that reconstructs nested dataclasses.
**Warning signs:** `convert` command crashes with AttributeError on record fields.

### Pitfall 4: Missing settings.json Default Path
**What goes wrong:** User runs `chuckles generate input.csv` without `--settings` and gets a confusing `FileNotFoundError: Settings file not found: settings.json`.
**Why it happens:** Default path "settings.json" resolves relative to cwd, which may not be the project root.
**How to avoid:** Use a clear default ("settings.json") but provide a helpful error message when the file is not found. The existing `load_config()` already raises `FileNotFoundError` with a clear message including the resolved path.
**Warning signs:** User confusion about where settings.json should be.

### Pitfall 5: Hub Push Failing Silently or Blocking
**What goes wrong:** `push_dataset()` fails because HF_TOKEN is not set, or the push takes a long time with no feedback.
**Why it happens:** The `--no-push` flag is forgotten, or the token is not exported.
**How to avoid:** Check for HF_TOKEN early (before generation starts) if push is enabled. Show a clear message about what is being pushed and where. The existing `push_dataset()` already raises `ValueError` for missing token.
**Warning signs:** Pipeline completes generation then crashes at the very end.

## Code Examples

### JSONL Round-Trip Deserialization
```python
# Source: hand-rolled based on types.py dataclass structure
import json
from pathlib import Path

from chuckles_prime.types import AgentTrace, GenerationRecord, ParodyCandidate


def _record_from_dict(d: dict) -> GenerationRecord:
    """Reconstruct a GenerationRecord from a dict (as produced by asdict + json)."""
    candidates = [
        ParodyCandidate(
            text=c["text"],
            phonetic_scores=c.get("phonetic_scores", {}),
            humor_note=c.get("humor_note", ""),
        )
        for c in d.get("candidates", [])
    ]
    trace_d = d.get("trace", {})
    trace = AgentTrace(
        steps=trace_d.get("steps", []),
        final_output=trace_d.get("final_output", ""),
        token_usage=trace_d.get("token_usage"),
        state=trace_d.get("state", "unknown"),
    )
    return GenerationRecord(
        input_title=d["input_title"],
        candidates=candidates,
        trace=trace,
        model_name=d.get("model_name", "unknown"),
        error=d.get("error"),
    )


def load_records(jsonl_path: Path) -> list[GenerationRecord]:
    """Load GenerationRecord objects from a JSONL file."""
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(_record_from_dict(json.loads(line)))
    return records
```

### Complete CLI Skeleton
```python
# Source: argparse official docs + rich official docs
import argparse
import sys
from pathlib import Path

from rich.console import Console

console = Console(stderr=True)  # Progress/status to stderr, data to stdout


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="chuckles",
        description="Generate phonetically sound parody titles and RLVR datasets",
    )
    sub = parser.add_subparsers(dest="command")

    # -- generate --
    gen = sub.add_parser("generate", help="Full pipeline: CSV -> parodies -> datasets -> Hub")
    gen.add_argument("input", help="Input CSV with 'title' column")
    gen.add_argument("--settings", default="settings.json",
                     help="Settings JSON path (default: settings.json)")
    gen.add_argument("--output-dir", default="output",
                     help="Output directory for traces (default: output/)")
    gen.add_argument("--grpo-repo", default=None,
                     help="HF Hub repo for GRPO dataset (e.g. user/chuckles-grpo)")
    gen.add_argument("--dpo-repo", default=None,
                     help="HF Hub repo for DPO dataset (e.g. user/chuckles-dpo)")
    gen.add_argument("--no-push", action="store_true",
                     help="Skip pushing datasets to Hub")
    gen.set_defaults(func=cmd_generate)

    # -- convert --
    conv = sub.add_parser("convert", help="Convert saved traces to datasets (no generation)")
    conv.add_argument("traces", help="Path to traces JSONL file")
    conv.add_argument("--settings", default="settings.json",
                     help="Settings JSON path (needed for DPO human examples)")
    conv.add_argument("--grpo-repo", default=None,
                     help="HF Hub repo for GRPO dataset")
    conv.add_argument("--dpo-repo", default=None,
                     help="HF Hub repo for DPO dataset")
    conv.add_argument("--no-push", action="store_true",
                     help="Skip pushing datasets to Hub")
    conv.set_defaults(func=cmd_convert)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    try:
        args.func(args)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)
```

### Settings Path Convention
```python
# Default: "settings.json" in current working directory
# Override: --settings /absolute/or/relative/path.json
# Resolution: load_config() already handles relative path resolution
#             relative to the settings file's own directory
```

### Output Directory Convention
```python
# Default: "output/" in current working directory
# Contains:
#   output/traces.jsonl         -- full generation records
# Override: --output-dir /path/to/dir
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
traces_path = output_dir / "traces.jsonl"
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `print()` with manual counters | `rich.progress.Progress` | rich already in deps | Professional progress display |
| `generate_batch()` with embedded prints | CLI calls `generate_single()` in loop | This phase | Decouples progress display from generation logic |
| No CLI, run as script | `chuckles generate` / `chuckles convert` | This phase | Entry point already in pyproject.toml |

**Not deprecated/not outdated:**
- argparse remains the standard stdlib choice for CLI parsing in Python 3.10+
- rich 14.x is current and stable
- `dataclasses.asdict()` is the correct serialization path for simple dataclasses

## Open Questions

1. **Hub repo ID defaults**
   - What we know: `push_dataset()` requires a `repo_id` string. User needs to specify where to push.
   - What's unclear: Should there be default repo IDs in settings.json, or should they always be CLI flags?
   - Recommendation: Make them CLI flags with no default. Require explicit `--grpo-repo` and `--dpo-repo` when `--no-push` is not set. This prevents accidental pushes to wrong repos.

2. **DPO dataset in convert command**
   - What we know: `build_dpo_dataset()` needs both `human_examples` (from config) and `model_records`. The convert command loads records from JSONL but also needs config for human examples.
   - What's unclear: Should `--settings` be required for convert, or optional (skip DPO if not provided)?
   - Recommendation: Make `--settings` required for convert. The DPO dataset is a core output. If the user only wants GRPO, they can omit `--dpo-repo`.

3. **Traces JSONL naming with timestamps**
   - What we know: `archive_traces()` writes to a single path. Running generate twice would overwrite.
   - What's unclear: Should output filenames include timestamps to prevent overwriting?
   - Recommendation: Use a fixed name `traces.jsonl` for simplicity. User can use `--output-dir` to separate runs. Overwrite is acceptable for a v1 CLI.

## Sources

### Primary (HIGH confidence)
- Existing codebase: `src/chuckles_prime/*.py` -- all module signatures and types read directly
- `pyproject.toml` -- entry point already declared, dependencies confirmed
- [Python argparse official documentation](https://docs.python.org/3/library/argparse.html) -- subparsers pattern
- [Rich Progress documentation](https://rich.readthedocs.io/en/latest/progress.html) -- Progress, track(), column types
- [Rich Tables documentation](https://rich.readthedocs.io/en/latest/tables.html) -- Table, add_column, add_row

### Secondary (MEDIUM confidence)
- [CLI comparison: argparse vs Click vs Typer](https://codecut.ai/comparing-python-command-line-interface-tools-argparse-click-and-typer/) -- confirmed argparse sufficiency for small CLIs
- [Dataclass serialization approaches](https://tomaugspurger.net/posts/serializing-dataclasses/) -- confirmed hand-rolled deserialization is viable for shallow nesting

### Tertiary (LOW confidence)
- None. All findings verified against codebase and official documentation.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- argparse is stdlib, rich is already installed, no new deps
- Architecture: HIGH -- all module APIs read directly from source, pipeline flow is clear
- Pitfalls: HIGH -- identified from direct code analysis (print conflicts, import weight, JSONL types)
- Code examples: HIGH -- verified against actual function signatures in the codebase

**Research date:** 2026-01-31
**Valid until:** 2026-03-31 (stable domain, no fast-moving dependencies)
