"""Application configuration loaded from external files.

All paths are resolved from a single settings JSON file.
No config paths are hardcoded in source code.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    """Immutable application configuration loaded from external files."""

    # LLM backend
    model_name: str
    api_base_url: str
    api_key_env_var: str

    # External data
    funny_words: dict[str, list[str]]
    preferences_text: str
    human_examples: list[tuple[str, str, str]]  # (input, output, explanation)

    # Paths (for reference)
    funny_words_path: Path
    preferences_path: Path
    human_examples_path: Path


_REQUIRED_SETTINGS_KEYS = {
    "funny_words_path",
    "preferences_path",
    "human_examples_path",
    "model_name",
    "api_base_url",
    "api_key_env_var",
}


def _resolve_path(path_str: str, base_dir: Path) -> Path:
    """Resolve a path relative to a base directory, expanding ~ and env vars."""
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return p.resolve()


def load_config(settings_path: str | Path) -> AppConfig:
    """Load application configuration from a settings JSON file.

    The settings file specifies paths to all external data files.
    Relative paths are resolved relative to the settings file's directory.

    Args:
        settings_path: Path to the settings JSON file.

    Returns:
        Populated frozen AppConfig instance.

    Raises:
        FileNotFoundError: If settings file or any referenced file is missing.
        ValueError: If settings JSON is missing required keys.
    """
    settings_path = Path(settings_path).expanduser().resolve()

    if not settings_path.exists():
        raise FileNotFoundError(f"Settings file not found: {settings_path}")

    with open(settings_path, encoding="utf-8") as f:
        settings = json.load(f)

    # Validate required keys
    missing = _REQUIRED_SETTINGS_KEYS - set(settings.keys())
    if missing:
        raise ValueError(
            f"Settings file missing required keys: {', '.join(sorted(missing))}"
        )

    base_dir = settings_path.parent

    # Resolve paths
    funny_words_path = _resolve_path(settings["funny_words_path"], base_dir)
    preferences_path = _resolve_path(settings["preferences_path"], base_dir)
    human_examples_path = _resolve_path(settings["human_examples_path"], base_dir)

    # Load funny_words.json
    if not funny_words_path.exists():
        raise FileNotFoundError(
            f"Funny words file not found: {funny_words_path} "
            f"(referenced in {settings_path})"
        )
    with open(funny_words_path, encoding="utf-8") as f:
        funny_words = json.load(f)

    # Load preferences.json
    if not preferences_path.exists():
        raise FileNotFoundError(
            f"Preferences file not found: {preferences_path} "
            f"(referenced in {settings_path})"
        )
    with open(preferences_path, encoding="utf-8") as f:
        prefs = json.load(f)
    preferences_text = prefs.get("style_description", "")

    # Load cleaned human examples CSV
    if not human_examples_path.exists():
        raise FileNotFoundError(
            f"Human examples file not found: {human_examples_path} "
            f"(referenced in {settings_path})"
        )
    human_examples: list[tuple[str, str, str]] = []
    with open(human_examples_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            human_examples.append((
                row.get("input", ""),
                row.get("output", ""),
                row.get("explanation", ""),
            ))

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
