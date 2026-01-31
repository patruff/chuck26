# Codebase Structure

**Analysis Date:** 2026-01-31

## Directory Layout

```
parodies2026/
├── .github/
│   └── workflows/
│       └── generate-parody.yml       # Automated GitHub Actions workflow (every 6 hours)
├── generate_parody.py                # Main parody generation script - single/CLI entry point
├── batch_generate.py                 # Batch processing from CSV files
├── drive_batch_processor.py           # Google Drive integration for automated workflows
├── test_popular_movies.py             # RLVR test suite with 10 sample movies
├── rlvr_dataset_tools.py              # Dataset labeling and format conversion tools
├── system_prompt.py                   # Centralized prompts and prompt building
├── word_structures.py                 # Custom phonetic pronunciations and funny word lists
├── word_phone.py                      # Phonetic analysis tool (uploaded to HF Hub)
├── parody_suggestions.py              # Word suggestion tool (uploaded to HF Hub)
├── upload_to_drive.py                 # Utility for Google Drive uploads
├── push_tool_to_hub.py                # Utility to push tools to Hugging Face Hub
├── known100.csv                       # 100 verified funny parody examples (training data)
├── input.csv                          # Sample input for batch processing
├── test.csv                           # Sample test file
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore patterns (excludes output, debug.log, env files)
├── README.md                          # Comprehensive documentation
└── .git/                              # Git repository metadata
```

## Directory Purposes

**`.github/workflows/`:**
- Purpose: GitHub Actions automation
- Contains: Workflow YAML files for scheduled/manual execution
- Key files: `generate-parody.yml` (main automation workflow)

**Root directory:**
- Purpose: Main entry points and core logic
- Contains: All Python scripts, config files, sample data
- Organization: One script per major responsibility

## Key File Locations

**Entry Points:**
- `generate_parody.py:main()`: Single title generation with CLI args
- `batch_generate.py:main()`: Batch CSV processing with CLI args
- `drive_batch_processor.py:main()`: Google Drive automated workflow
- `test_popular_movies.py:main()`: RLVR dataset generation from 10 test movies

**Configuration:**
- `system_prompt.py`: All prompts (agent system prompt, style guide, generation template)
- `word_structures.py`: Funny word lists, custom phonetic pronunciations, known parodies
- `requirements.txt`: Python package dependencies

**Core Logic:**
- `generate_parody.py:generate_parody()`: Core generation function using smolagents CodeAgent
- `generate_parody.py:CerebrasModel`: Cerebras API adapter for smolagents
- `generate_parody.py:OutputCapture`: Output management and debugging
- `word_phone.py`: Phonetic similarity scoring algorithm
- `parody_suggestions.py`: Word suggestion matching algorithm

**Testing & Data:**
- `test_popular_movies.py`: End-to-end test with 10 movies + RLVR capture
- `known100.csv`: Training data (original, parody, reasoning)
- `input.csv`: Sample batch input
- `test.csv`: Sample test input

**Utilities:**
- `rlvr_dataset_tools.py`: Dataset labeling, format conversion (SFT/DPO/RLVR)
- `batch_generate.py`: Wrapper for batch CSV processing
- `drive_batch_processor.py`: Google Drive integration (finds, downloads, uploads files)
- `upload_to_drive.py`: Standalone Google Drive upload utility
- `push_tool_to_hub.py`: Push tools to Hugging Face Hub

**Output Directories (created at runtime):**
- `output/`: CSV results and raw output files (created by `generate_parody.py`)
- `parody_output/`: Debug files per step (created by `OutputCapture`)
- `rlvr_output/`: RLVR dataset files (created by `test_popular_movies.py`)
- `.parody_processor/`: State tracking for Google Drive workflow

## Naming Conventions

**Files:**
- Snake case with `.py` extension: `generate_parody.py`, `word_structures.py`
- Descriptive names matching primary responsibility
- CLI entry points named by action: `generate_parody`, `batch_generate`, `drive_batch_processor`
- Utility/tool files named by purpose: `rlvr_dataset_tools`, `word_phone`, `parody_suggestions`

**Directories:**
- Lowercase: `.github/workflows/`, `output/`, `rlvr_output/`
- Purpose is obvious from name: `workflows/` for CI/CD, output dirs named after their content

**Functions:**
- Snake case: `generate_parody()`, `extract_data()`, `get_drive_service()`
- Private/internal functions start with `_`: `_preprocess_content()`, `_get_word_phones()`

**Classes:**
- Pascal case: `CerebrasModel`, `OutputCapture`, `RLVRTemplateTags`, `ParodyWordSuggestionTool`
- Names match their responsibility

**Variables:**
- Snake case: `funny_words`, `custom_phones`, `suggestions_json`, `processed_state`
- Constants: UPPER_SNAKE_CASE: `GENERATION_PROMPT_TEMPLATE`, `AGENT_SYSTEM_PROMPT`, `DRIVE_BASE_FOLDER`

**CSV Columns:**
- Lowercase with underscores: `original`, `parody`, `reasoning`, `title`, `input`, `parody_result`

## Where to Add New Code

**New Feature (e.g., custom word categories):**
- Primary code: `word_structures.py:FUNNY_WORDS_BY_CATEGORY`
  - Add new category dict with word lists
  - Reference in `generate_parody.py:generate_parody()` where funny_words are used
- Tests: Add test movie to `test_popular_movies.py:TEST_MOVIES`
- Configuration: If new custom pronunciations needed, add to `word_structures.py:custom_phones`

**New Generation Strategy (e.g., multi-word parody algorithm):**
- Implementation: New function in `generate_parody.py` or separate `parody_strategies.py`
- Integration: Modify `GENERATION_PROMPT_TEMPLATE` in `system_prompt.py` to guide agent
- Testing: Add test case to `test_popular_movies.py` with flag like `--strategy multi-word`

**New Output Format (e.g., JSON instead of CSV):**
- Implement: New class in `generate_parody.py` extending `OutputCapture`
  - Or add method to `OutputCapture` for different export formats
- Integration: Add flag to `batch_generate.py:main()` like `--output-format json`
- Files affected: Only `generate_parody.py`, `batch_generate.py`

**New Data Processing Tool (e.g., dataset filtering):**
- Location: Add new subcommand to `rlvr_dataset_tools.py`
- Pattern: Copy `auto_label_command()` structure, modify logic
- File I/O: Follow existing pattern of read JSONL → process → write JSONL

**New External Integration (e.g., Anthropic Claude instead of Cerebras):**
- Implementation: New adapter class in `generate_parody.py` (e.g., `ClaudeModel`)
- Must implement: `__call__(messages, stop_sequences, **kwargs)` interface
- Integration: Pass via `model_class` param or CLI flag `--backend claude`
- No changes needed to: `smolagents`, prompt system, tool definitions

**New Tool (e.g., rhythm analyzer):**
- Implementation: New Tool subclass in `rhythm_analyzer.py`
- Inheritance: From `smolagents.Tool` (see `word_phone.py` for pattern)
- Registration: Load in `generate_parody.py`: `load_tool("patruff/rhythm-analyzer")`
- Usage: Add to tools list in `CodeAgent(tools=[...])`

## Special Directories

**`.github/workflows/`:**
- Purpose: GitHub Actions CI/CD configuration
- Generated: No (manually created)
- Committed: Yes
- Trigger: Schedule (every 6 hours) or manual dispatch
- Key workflow: `generate-parody.yml` runs `drive_batch_processor.py`

**`output/` (created at runtime):**
- Purpose: CSV results and raw output from single/batch generation
- Generated: Yes (by `OutputCapture.export_to_csv()`)
- Committed: No (in `.gitignore`)
- Created by: `generate_parody.py`, `batch_generate.py`
- Format: CSV with columns `input,parody_result,reasoning,timestamp`

**`parody_output/` (created at runtime):**
- Purpose: Per-step debug files from agent execution
- Generated: Yes (by `OutputCapture.callback()`)
- Committed: No (in `.gitignore`)
- Created by: `generate_parody.py` during agent loop
- Files:
  - `llm_output_*.txt`: Raw LLM response at each step
  - `full_dump_*.txt`: All step log attributes
  - `action_output_*.txt`: Tool call results
  - `RAW_*.txt`: Complete execution transcript

**`rlvr_output/` (created at runtime):**
- Purpose: RLVR training dataset files
- Generated: Yes (by `test_popular_movies.py`)
- Committed: No (in `.gitignore`)
- Structure:
  - `datasets/`: JSONL dataset files (full, tool-only, reasoning-only)
  - `individual/`: Per-movie debug output
- Created by: `test_popular_movies.py --output-dir rlvr_output`

**`.parody_processor/` (user home directory):**
- Purpose: State tracking for Google Drive workflow
- Generated: Yes (by `drive_batch_processor.py`)
- Committed: No (local-only state)
- File: `processed_state.json` maps `{file_id: timestamp}`
- Used for: Deduplication when workflow runs multiple times

## Dependency Flow

```
generate_parody.py (main)
├── Imports: system_prompt, word_structures, smolagents, cerebras
├── Classes: CerebrasModel, OutputCapture
├── Functions: generate_parody()
└── Called by: batch_generate.py, drive_batch_processor.py, test_popular_movies.py

batch_generate.py
├── Imports: generate_parody
├── Reads: input.csv
├── Calls: generate_parody() for each title
└── Writes: output.csv

drive_batch_processor.py
├── Imports: generate_parody, google-api libraries
├── Uses: Google Drive API
├── Calls: batch-like processing via generate_parody()
└── Uploads results to: Google Drive parodiesdata/output/

test_popular_movies.py
├── Imports: generate_parody
├── Calls: generate_parody() for 10 test movies
└── Captures: RLVR structured data

rlvr_dataset_tools.py
├── No imports of other project modules
├── Reads: JSONL datasets (output of test_popular_movies.py)
├── Processes: Labeling, format conversion
└── Writes: Formatted training data

system_prompt.py
├── Pure configuration (no imports of other project modules)
├── Imported by: generate_parody.py, (optionally) test_popular_movies.py
└── Provides: All prompt strings, template building

word_structures.py
├── Pure configuration + CSV loading
├── Imported by: generate_parody.py
└── Provides: funny_words list, custom_phones dict, known parodies
```

## Python Module Organization

**No packages (src/ structure):**
- All modules at root level
- Flat import structure: `from word_structures import custom_phones`
- Advantage: Simplicity for scripts and utilities
- Trade-off: Not suitable for large projects, but adequate for focused tool

**Module Sizes (approximate):**
- `generate_parody.py`: ~460 lines (largest, contains main logic + classes)
- `rlvr_dataset_tools.py`: ~450 lines (dataset processing)
- `test_popular_movies.py`: ~400 lines (RLVR testing)
- `parody_suggestions.py`: ~280 lines (tool implementation)
- `word_phone.py`: ~280 lines (tool implementation)
- `system_prompt.py`: ~190 lines (all prompts)
- Others: <150 lines each

## Configuration Management

**Environment Variables:**
- `CEREBRAS_API_KEY`: Cerebras API authentication (required)
- `GOOGLE_DRIVE_CREDENTIALS`: JSON service account key (required for Drive workflow)
- `CEREBRAS_MODEL`: Optional model override (used by GitHub Actions)

**CLI Arguments:**
- `--title`: Movie title to parody (default: "The Running Man")
- `--model`: Cerebras model name (default: "qwen-3-32b")
- `--output-dir`: Output directory path (default: "./output")
- `--input`: Input CSV file (default: "input.csv")
- `--output`: Output CSV file (default: "output.csv")
- `--limit`: Limit number of test movies (default: None, use all)

**Hard-coded Configuration:**
- `PHONETIC_SCORE_THRESHOLD`: 0.6 (in prompts and tools)
- `MIN_HUMOR_RATING`: 6 (in RLVR auto-labeling)
- `MIN_TOOL_CALLS`: 2 (in RLVR quality criteria)
- Model default: "qwen-3-32b"

---

*Structure analysis: 2026-01-31*
