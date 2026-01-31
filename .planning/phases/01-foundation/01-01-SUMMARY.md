---
phase: 01-foundation
plan: 01
subsystem: config
tags: [python-package, csv-cleaning, config-loading, dataclass]
dependency-graph:
  requires: []
  provides: [installable-package, config-system, csv-cleaner, AppConfig]
  affects: [01-02, 02-01, 03-01, 04-01]
tech-stack:
  added: [smolagents, openai, rich, pronouncing, pytest]
  patterns: [frozen-dataclass-config, settings-json-indirection, csv-zone-parser]
key-files:
  created:
    - pyproject.toml
    - src/chuckles_prime/__init__.py
    - src/chuckles_prime/py.typed
    - src/chuckles_prime/config.py
    - src/chuckles_prime/csv_cleaner.py
    - tests/__init__.py
    - tests/test_config.py
  modified: []
decisions:
  - id: setuptools-build
    context: "Build system selection for pyproject.toml"
    choice: "setuptools with setuptools.build_meta backend"
    reason: "Standard, widely supported, no additional tooling needed"
  - id: frozen-dataclass
    context: "Config immutability pattern"
    choice: "frozen=True dataclass for AppConfig"
    reason: "Prevents accidental mutation of config at runtime"
  - id: csv-zone-parser
    context: "human_parodies.csv has 5 distinct format zones"
    choice: "Line-by-line zone detection with regex fallback to standard CSV"
    reason: "CSV reader alone cannot handle markdown and slash-separated formats"
  - id: settings-indirection
    context: "How to reference external config files"
    choice: "Single settings.json with absolute/relative paths to all data files"
    reason: "No hardcoded paths in source; all paths flow through one settings file"
metrics:
  duration: "5m 29s"
  completed: "2026-01-31"
---

# Phase 01 Plan 01: Package Skeleton & Config System Summary

**Installable Python package with frozen AppConfig, settings-JSON-driven config loading, and 5-zone CSV cleaner producing 1098 unique cleaned parody examples.**

## What Was Built

### Task 1: Package Skeleton (5d70b90)
- Created `pyproject.toml` with setuptools build system, project metadata, and dependencies (smolagents, openai, rich, pronouncing)
- Created `src/chuckles_prime/__init__.py` with `__version__ = "0.1.0"`
- Added `py.typed` marker for type checking
- Forward-declared CLI entry point (`chuckles = chuckles_prime.cli:main`)
- Verified: `pip install -e .` succeeds and `import chuckles_prime` works

### Task 2: CSV Cleaner, AppConfig, and Tests (940a544)

**CSV Cleaner** (`csv_cleaner.py`):
- Handles all 5 format zones in `human_parodies.csv`:
  - Zone 1 (rows 2-13): Standard 3-column CSV
  - Zone 2 (rows 14-24): Markdown-numbered `N. **input,output**` (comma splits across CSV fields)
  - Zone 3 (rows 25-68): Slash-separated `input / output: Explanation: ...`
  - Zone 4 (rows 69-~500): Triple-quoted with `Parody: ""output""**`
  - Zone 5 (rows ~501+): Standard 3-column CSV again
- Fixes encoding artifacts: `A(c)` to e-acute, `AY=` to a-ring
- Strips markdown `**`, normalizes whitespace, removes stray quotes
- Deduplicates by lowercase `(input, output)` pair
- Skips rows where input == output (no actual parody)
- Produces 1098 unique rows from 1234 raw lines

**AppConfig** (`config.py`):
- Frozen dataclass with fields for LLM backend (model_name, api_base_url, api_key_env_var), external data (funny_words, preferences_text, human_examples), and resolved paths
- `load_config(settings_path)` factory function reads settings JSON and loads all referenced files
- All paths resolved relative to settings file parent directory
- Clear error messages for missing files and missing keys

**Tests** (`tests/test_config.py`):
- `test_load_config_happy_path`: Full round-trip with temp files, verifies all fields, confirms frozen
- `test_load_config_missing_file`: FileNotFoundError for missing funny_words.json
- `test_load_config_missing_key`: ValueError for missing model_name
- `test_clean_human_parodies`: Tests first 80 lines of real CSV, validates no artifacts
- `test_clean_deduplication`: Verifies duplicate removal works correctly

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed Zone 2 CSV field splitting**
- **Found during:** Task 2, Step A
- **Issue:** The comma in `**Chinchilla,Chintrilla**` causes csv.reader to split Zone 2 rows into separate fields. The original regex `^\d+\.\s*\*\*(.+?),\s*(.+?)\*\*$` could never match `first_field` alone.
- **Fix:** Replaced single-regex approach with two-part detection: `zone2_first_re` matches `^\d+\.\s*\*\*(.+)$` on field[0], `zone2_second_re` matches `^(.+?)\*\*$` on field[1]. Both must match for Zone 2 classification.
- **Files modified:** `src/chuckles_prime/csv_cleaner.py`
- **Commit:** 940a544

**2. [Rule 3 - Blocking] Fixed pyproject.toml build-backend path**
- **Found during:** Task 1
- **Issue:** Initial build-backend `setuptools.backends._legacy:_Backend` does not exist in current setuptools versions, causing `pip install -e .` to fail with `BackendUnavailable`.
- **Fix:** Changed to `setuptools.build_meta` which is the standard backend.
- **Files modified:** `pyproject.toml`
- **Commit:** 5d70b90

## Verification Results

| Check | Result |
|-------|--------|
| `pip install -e .` | Pass |
| `import chuckles_prime` version check | 0.1.0 |
| All module imports | Pass |
| `pytest tests/test_config.py -v` | 5/5 passed |
| Full CSV clean (1234 raw lines) | 1098 unique rows |
| No encoding artifacts in output | Confirmed |
| No broken formatting in output | Confirmed |

## Next Phase Readiness

- Package is installable and importable
- Config system is ready to serve all subsequent phases
- CSV cleaner output (1098 rows) available for training data pipeline
- No blockers for Phase 01 Plan 02 (LLM adapter)
