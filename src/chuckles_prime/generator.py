"""Core generation engine orchestrating CodeAgent for parody title generation.

Wires together types, tools, and prompts with smolagents CodeAgent to
process titles from CSV input into structured GenerationRecord output.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from smolagents import CodeAgent

from chuckles_prime.prompts import PARODY_INSTRUCTIONS, build_generation_prompt
from chuckles_prime.tools import pre_compute_suggestions
from chuckles_prime.types import AgentTrace, GenerationRecord, ParodyCandidate

if TYPE_CHECKING:
    from smolagents import Tool

    from chuckles_prime.config import AppConfig


def read_input_titles(csv_path: str | Path) -> list[str]:
    """Read titles from a CSV file with a 'title' column header.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        List of title strings with whitespace stripped, empty rows skipped.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the CSV has no 'title' column.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    titles: list[str] = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "title" not in reader.fieldnames:
            raise ValueError(
                f"CSV file {csv_path} must have a 'title' column header"
            )
        for row in reader:
            title = row.get("title", "").strip()
            if title:
                titles.append(title)
    return titles


def create_agent(model: Any, parody_tool: Tool, phone_tool: Tool) -> CodeAgent:
    """Create a smolagents CodeAgent configured for parody generation.

    Args:
        model: OpenAICompatibleModel instance.
        parody_tool: Loaded parody_word_suggester tool (not passed to agent).
        phone_tool: Loaded word_phonetic_analyzer tool (passed to agent).

    Returns:
        Configured CodeAgent ready for parody generation.
    """
    return CodeAgent(
        tools=[phone_tool],
        model=model,
        instructions=PARODY_INSTRUCTIONS,
        additional_authorized_imports=["json"],
        return_full_result=True,
        max_steps=15,
    )


def _parse_agent_output(raw_output: str) -> list[ParodyCandidate]:
    """Parse agent output into ParodyCandidate objects.

    Handles valid JSON, JSON wrapped in explanation text, and invalid output.

    Args:
        raw_output: Raw string from agent's final_answer().

    Returns:
        List of ParodyCandidate objects (0-2 items).
    """
    data = None

    # Try direct JSON parse
    try:
        data = json.loads(raw_output)
    except (json.JSONDecodeError, TypeError):
        pass

    # Try extracting JSON from wrapper text
    if data is None and raw_output:
        first_brace = raw_output.find("{")
        last_brace = raw_output.rfind("}")
        if first_brace != -1 and last_brace > first_brace:
            try:
                data = json.loads(raw_output[first_brace : last_brace + 1])
            except (json.JSONDecodeError, TypeError):
                pass

    if data is None or not isinstance(data, dict):
        return []

    # Build attempts lookup for phonetic_scores and humor_note
    attempts_lookup: dict[str, dict[str, Any]] = {}
    for attempt in data.get("attempts", []):
        if isinstance(attempt, dict) and "text" in attempt:
            attempts_lookup[attempt["text"]] = attempt

    candidates: list[ParodyCandidate] = []
    for key in ("parody1", "parody2"):
        text = data.get(key)
        if not text or not isinstance(text, str):
            continue

        # Match attempt data if available
        attempt_data = attempts_lookup.get(text, {})
        raw_scores = attempt_data.get("scores", {})
        humor = attempt_data.get("humor_note", "")

        # Coerce scores to float, extracting numbers from strings if needed
        scores: dict[str, float] = {}
        if isinstance(raw_scores, dict):
            for k, v in raw_scores.items():
                try:
                    scores[k] = float(v)
                except (TypeError, ValueError):
                    # Try to extract a number from strings like "Custom (0.8 ...)"
                    nums = re.findall(r"(\d+\.?\d*)", str(v))
                    if nums:
                        scores[k] = float(nums[0])

        candidates.append(
            ParodyCandidate(
                text=text,
                phonetic_scores=scores,
                humor_note=humor if isinstance(humor, str) else "",
            )
        )

    return candidates[:2]


def _extract_trace(result: Any) -> AgentTrace:
    """Extract reasoning trace from a smolagents RunResult.

    Args:
        result: RunResult object from agent.run().

    Returns:
        Populated AgentTrace with steps, output, token usage, and state.
    """
    # Extract steps
    steps = []
    if hasattr(result, "steps") and result.steps:
        for step in result.steps:
            if isinstance(step, dict):
                steps.append(step)
            elif hasattr(step, "__dict__"):
                steps.append(step.__dict__)
            else:
                steps.append({"raw": str(step)})

    # Extract final output
    final_output = str(result.output) if hasattr(result, "output") else ""

    # Extract token usage
    token_usage = None
    if hasattr(result, "token_usage") and result.token_usage is not None:
        tu = result.token_usage
        if hasattr(tu, "input_tokens") and hasattr(tu, "output_tokens"):
            token_usage = {
                "input_tokens": tu.input_tokens,
                "output_tokens": tu.output_tokens,
            }
        elif isinstance(tu, dict):
            token_usage = tu

    # Extract state
    state = "success"
    if hasattr(result, "state"):
        state = str(result.state) if result.state else "success"

    return AgentTrace(
        steps=steps,
        final_output=final_output,
        token_usage=token_usage,
        state=state,
    )


def generate_single(
    title: str,
    agent: CodeAgent,
    parody_tool: Tool,
    config: AppConfig,
) -> GenerationRecord:
    """Generate parody candidates for a single title.

    Args:
        title: Input title string.
        agent: Configured CodeAgent.
        parody_tool: Loaded parody_word_suggester for pre-computation.
        config: Application configuration.

    Returns:
        GenerationRecord with candidates, trace, and optional error.
    """
    # Pre-compute suggestions
    suggestions = pre_compute_suggestions(title, config.funny_words, parody_tool)

    # Build prompt
    prompt = build_generation_prompt(
        title, suggestions, config.human_examples, config.preferences_text
    )

    # Run agent
    result = agent.run(task=prompt, reset=True)

    # Parse output and extract trace
    raw_output = str(result.output) if hasattr(result, "output") else str(result)
    candidates = _parse_agent_output(raw_output)
    trace = _extract_trace(result)

    # Check for max_steps_error
    error = None
    if trace.state == "max_steps_error":
        error = "max_steps_error: agent did not reach final_answer"

    return GenerationRecord(
        input_title=title,
        candidates=candidates,
        trace=trace,
        model_name=agent.model.model_id if hasattr(agent.model, "model_id") else str(agent.model),
        error=error,
    )


def generate_batch(
    titles: list[str],
    agent: CodeAgent,
    parody_tool: Tool,
    config: AppConfig,
) -> list[GenerationRecord]:
    """Generate parody candidates for a batch of titles.

    Isolates errors per-title so one failure does not crash the batch.

    Args:
        titles: List of input title strings.
        agent: Configured CodeAgent.
        parody_tool: Loaded parody_word_suggester for pre-computation.
        config: Application configuration.

    Returns:
        List of GenerationRecord objects (successes and failures).
    """
    records: list[GenerationRecord] = []

    for i, title in enumerate(titles):
        try:
            record = generate_single(title, agent, parody_tool, config)
        except Exception as e:
            record = GenerationRecord(
                input_title=title,
                candidates=[],
                trace=AgentTrace(
                    steps=[], final_output="", token_usage=None, state="error"
                ),
                model_name=agent.model.model_id
                if hasattr(agent.model, "model_id")
                else str(agent.model),
                error=str(e),
            )

        print(
            f"[{i + 1}/{len(titles)}] {title} ... "
            f"{'OK' if not record.error else 'ERROR: ' + record.error}"
        )
        records.append(record)

    return records
