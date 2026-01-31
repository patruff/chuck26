"""Data types for generation output.

Defines the structured containers that flow through the generation pipeline:
ParodyCandidate -> AgentTrace -> GenerationRecord.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParodyCandidate:
    """A single parody candidate with quality signals."""

    text: str
    phonetic_scores: dict[str, float]  # original_word -> similarity_score
    humor_note: str = ""


@dataclass
class AgentTrace:
    """Captured reasoning trace from one CodeAgent run."""

    steps: list[dict[str, Any]]  # Raw step dicts from RunResult.steps
    final_output: str  # Raw string from final_answer()
    token_usage: dict[str, int] | None  # {input_tokens, output_tokens} or None
    state: str  # "success" or "max_steps_error" or "error"


@dataclass
class GenerationRecord:
    """One generation result, ready for downstream dataset conversion."""

    input_title: str
    candidates: list[ParodyCandidate]  # Target: 2 candidates
    trace: AgentTrace
    model_name: str
    error: str | None = None  # Non-None if generation failed
