"""Tests for config loading and CSV cleaning."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from chuckles_prime.config import AppConfig, load_config
from chuckles_prime.csv_cleaner import clean_human_parodies

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_FUNNY_WORDS = {
    "body_parts": ["butt", "toenail"],
    "animals": ["platypus", "wombat"],
}

SAMPLE_PREFERENCES = {
    "style_description": "Make it absurd and punny.",
    "other_key": "ignored",
}

SAMPLE_CSV_ROWS = [
    ("Wolverine", "Pullverine", "Funny replacement of Wol with Pull"),
    ("Diablo", "Peeablo", "Childish prefix"),
    ("Tom Cruise", "Tom Poos", "Juvenile reference"),
    ("Spartacus", "Fartacus", "Flatulence humor"),
    ("Mozart", "Mozfart", "Fart replacement"),
]


def _write_settings(
    tmp_path: Path,
    *,
    funny_words: dict | None = None,
    preferences: dict | None = None,
    csv_rows: list[tuple[str, str, str]] | None = None,
    extra_settings: dict | None = None,
    omit_keys: set[str] | None = None,
) -> Path:
    """Create a complete settings environment in tmp_path and return settings.json path."""
    fw_path = tmp_path / "funny_words.json"
    pref_path = tmp_path / "preferences.json"
    csv_path = tmp_path / "examples.csv"

    # Write funny words
    fw_data = funny_words if funny_words is not None else SAMPLE_FUNNY_WORDS
    fw_path.write_text(json.dumps(fw_data), encoding="utf-8")

    # Write preferences
    pref_data = preferences if preferences is not None else SAMPLE_PREFERENCES
    pref_path.write_text(json.dumps(pref_data), encoding="utf-8")

    # Write CSV
    rows = csv_rows if csv_rows is not None else SAMPLE_CSV_ROWS
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["input", "output", "explanation"], quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        for inp, out, expl in rows:
            writer.writerow({"input": inp, "output": out, "explanation": expl})

    # Write settings
    settings = {
        "funny_words_path": str(fw_path),
        "preferences_path": str(pref_path),
        "human_examples_path": str(csv_path),
        "model_name": "qwen-3-32b",
        "api_base_url": "https://api.cerebras.ai/v1",
        "api_key_env_var": "CEREBRAS_API_KEY",
    }
    if extra_settings:
        settings.update(extra_settings)
    if omit_keys:
        for key in omit_keys:
            settings.pop(key, None)

    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps(settings), encoding="utf-8")
    return settings_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_load_config_happy_path(tmp_path: Path) -> None:
    """load_config returns fully populated AppConfig from valid settings."""
    settings_path = _write_settings(tmp_path)
    config = load_config(settings_path)

    # Type check
    assert isinstance(config, AppConfig)

    # LLM fields
    assert config.model_name == "qwen-3-32b"
    assert config.api_base_url == "https://api.cerebras.ai/v1"
    assert config.api_key_env_var == "CEREBRAS_API_KEY"

    # Funny words
    assert config.funny_words == SAMPLE_FUNNY_WORDS
    assert "body_parts" in config.funny_words
    assert "platypus" in config.funny_words["animals"]

    # Preferences
    assert config.preferences_text == "Make it absurd and punny."

    # Human examples
    assert len(config.human_examples) == len(SAMPLE_CSV_ROWS)
    assert config.human_examples[0] == (
        "Wolverine", "Pullverine", "Funny replacement of Wol with Pull",
    )

    # Paths are resolved Path objects
    assert isinstance(config.funny_words_path, Path)
    assert config.funny_words_path.exists()

    # Frozen -- cannot mutate
    with pytest.raises(FrozenInstanceError):
        config.model_name = "something-else"  # type: ignore[misc]


def test_load_config_missing_file(tmp_path: Path) -> None:
    """load_config raises FileNotFoundError when a referenced file is missing."""
    settings_path = _write_settings(tmp_path)

    # Delete funny_words.json to simulate missing file
    (tmp_path / "funny_words.json").unlink()

    with pytest.raises(FileNotFoundError, match="[Ff]unny"):
        load_config(settings_path)


def test_load_config_missing_key(tmp_path: Path) -> None:
    """load_config raises ValueError when required settings key is absent."""
    settings_path = _write_settings(tmp_path, omit_keys={"model_name"})

    with pytest.raises(ValueError, match="model_name"):
        load_config(settings_path)


def test_clean_human_parodies(tmp_path: Path) -> None:
    """CSV cleaner produces valid output from real data (first 80 lines)."""
    real_csv = Path(
        "/Users/patruff/chucklesPRIME/parodies2026/human_parodies/human_parodies.csv"
    )
    if not real_csv.exists():
        pytest.skip("human_parodies.csv not found at expected path")

    # Copy first 80 lines to a temp file
    raw_lines = real_csv.read_text(encoding="utf-8", errors="replace").splitlines()
    subset = "\n".join(raw_lines[:80])
    temp_input = tmp_path / "raw_subset.csv"
    temp_input.write_text(subset, encoding="utf-8")

    temp_output = tmp_path / "cleaned.csv"
    count = clean_human_parodies(temp_input, temp_output)

    assert temp_output.exists()
    assert count > 0

    # Read output and validate
    with open(temp_output, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == count

    for row in rows:
        # All 3 columns must be present
        assert "input" in row
        assert "output" in row
        assert "explanation" in row

        # Input and output must be non-empty
        assert row["input"].strip(), f"Empty input in row: {row}"
        assert row["output"].strip(), f"Empty output in row: {row}"

        # No encoding artifacts
        assert "A(c)" not in row["input"], f"Encoding artifact in input: {row['input']}"
        assert "A(c)" not in row["output"], f"Encoding artifact in output: {row['output']}"
        assert "A(c)" not in row["explanation"], f"Encoding artifact in explanation"

        # No markdown ** artifacts
        assert "**" not in row["input"], f"Markdown artifact in input: {row['input']}"
        assert "**" not in row["output"], f"Markdown artifact in output: {row['output']}"


def test_clean_deduplication(tmp_path: Path) -> None:
    """CSV cleaner deduplicates rows with same input/output pair."""
    # Create CSV with duplicates (varying case and whitespace)
    temp_input = tmp_path / "dupes.csv"
    with open(temp_input, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["input", "output", "explanation"])
        writer.writerow(["Wolverine", "Pullverine", "First occurrence"])
        writer.writerow(["wolverine", "pullverine", "Duplicate (different case)"])
        writer.writerow(["Wolverine ", " Pullverine", "Duplicate (whitespace)"])
        writer.writerow(["Diablo", "Peeablo", "Unique row"])
        writer.writerow(["Diablo", "Peeablo", "Another dupe of Diablo"])

    temp_output = tmp_path / "deduped.csv"
    count = clean_human_parodies(temp_input, temp_output)

    # Should have exactly 2 unique rows: Wolverine/Pullverine and Diablo/Peeablo
    assert count == 2

    with open(temp_output, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 2
    assert rows[0]["input"] == "Wolverine"
    assert rows[0]["explanation"] == "First occurrence"
    assert rows[1]["input"] == "Diablo"
