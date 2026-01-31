# Architecture

**Analysis Date:** 2026-01-31

## Pattern Overview

**Overall:** Agent-Orchestrated AI Pipeline with Tool Integration

The parody generator uses Hugging Face's **smolagents** CodeAgent framework to orchestrate phonetic analysis and parody generation. The LLM (Cerebras) acts as the decision-maker, calling specialized phonetic tools to verify suggestions before finalizing parodies.

**Key Characteristics:**
- Tool-augmented LLM reasoning with structured prompt guidance
- Two-phase generation: suggestions → verification → synthesis
- Modular phonetic analysis layer for phoneme-based similarity scoring
- RLVR dataset capture for training data generation
- Google Drive integration for batch processing and data management

## Layers

**LLM Orchestration Layer:**
- Purpose: Central decision-making and parody generation using CodeAgent
- Location: `generate_parody.py` (CerebrasModel class, CodeAgent setup)
- Contains: Agent initialization, message formatting, Cerebras API integration
- Depends on: Cerebras SDK, smolagents framework, system prompts
- Used by: Main generation pipeline, batch processors

**Tool Integration Layer:**
- Purpose: Provide phonetic verification and word suggestion capabilities
- Location: `word_phone.py`, `parody_suggestions.py` (Hugging Face Hub tools)
- Contains: CMU dictionary phoneme lookup, Levenshtein distance calculation, vowel grouping
- Depends on: `pronouncing` library, custom pronunciations from `word_structures.py`
- Used by: CodeAgent during reasoning phase

**Configuration & Prompt Layer:**
- Purpose: Centralize all prompts, examples, and configuration
- Location: `system_prompt.py`, `word_structures.py`
- Contains: Agent system prompts, generation templates, known parody examples, funny word lists, custom phonetic pronunciations
- Depends on: CSV files (`known100.csv`), runtime configuration
- Used by: Generation pipeline for prompt construction

**Data Processing Layer:**
- Purpose: Handle batch processing, RLVR dataset formatting, and output management
- Location: `batch_generate.py`, `test_popular_movies.py`, `rlvr_dataset_tools.py`, `drive_batch_processor.py`
- Contains: CSV reading/writing, dataset labeling, format conversion (SFT/DPO/RLVR), Google Drive I/O
- Depends on: `generate_parody.py`, file I/O, Google Drive API
- Used by: Batch workflows, RLVR training preparation

**Output Capture Layer:**
- Purpose: Capture and organize model outputs for debugging and DPO training
- Location: `generate_parody.py` (OutputCapture class)
- Contains: File writing, CSV export, raw output capture, data extraction
- Depends on: File system, regex-based text parsing
- Used by: Generation pipeline for every step callback

## Data Flow

**Single Title Generation Flow:**

1. **Input Processing** (`generate_parody.py:main()`)
   - CLI args parsed for title, model, output directory
   - Environment variable `CEREBRAS_API_KEY` validated

2. **Suggestion Generation** (`generate_parody.py:generate_parody()`)
   - Title split into words
   - For each word: `parody_tool.forward()` called with word list from `word_structures.py:funny_words`
   - Parody suggestions returned with phonetic similarity scores
   - Results stored as JSON mapping: `{word: [{suggestion, score}, ...]}`

3. **Prompt Construction** (`system_prompt.py:build_generation_prompt()`)
   - Known funny parodies loaded from `known100.csv` as examples
   - Generation template from `GENERATION_PROMPT_TEMPLATE` populated with:
     - Title to parody
     - Style guide (`PARODY_STYLE_GUIDE`)
     - Pre-computed suggestions
     - Instructions for verification step

4. **Agent Initialization & Reasoning** (`generate_parody.py:generate_parody()`)
   - `CerebrasModel` created with API key
   - `CodeAgent` created with:
     - `word_phone_tool` for verification
     - System prompt from `AGENT_SYSTEM_PROMPT`
     - Cerebras model backend
   - `agent.run(prompt)` executes reasoning loop

5. **Agent Iteration Loop** (smolagents internally)
   - Agent reads prompt with suggestions
   - Generates multiple parody attempts
   - Calls `word_phone_tool()` for each attempt to verify scores
   - Reasoning captured in `<think>` tags
   - Continues until completion

6. **Output Capture** (`OutputCapture.callback()`)
   - Each agent step: `llm_output` extracted and saved to files
   - Pattern matching extracts final parody and reasoning
   - CSV file generated: `{number}_{timestamp}_{model}.csv`

7. **Result Return**
   - Final parody string returned
   - Raw output saved to `RAW_{title}_{timestamp}.txt`

**Batch Processing Flow:**

1. **CSV Input Reading** (`batch_generate.py:process_batch()`)
   - Read `input.csv` with "title" column

2. **Per-Title Processing**
   - Call `generate_parody()` for each title
   - Extract parody using regex pattern matching
   - Accumulate results

3. **CSV Output Writing**
   - Write results to `output.csv` with columns: `input`, `parody_result`, `reasoning`

**Google Drive Automated Workflow:**

1. **File Discovery** (`drive_batch_processor.py:main()`)
   - Connect to Google Drive using service account credentials
   - Find/create `parodiesdata` folder structure
   - List files in `parodiesdata/input/` folder

2. **File Download & Processing**
   - Download each CSV from input folder
   - Process using `batch_generate.py` pipeline
   - Generate parodies

3. **File Upload**
   - Create output filename: `output_{original_filename}_{timestamp}.csv`
   - Upload to `parodiesdata/output/` folder
   - Track processed files to avoid duplicates

**RLVR Dataset Capture Flow:**

1. **Movie List Iteration** (`test_popular_movies.py`)
   - 10 predefined movies: Matrix, Die Hard, Fight Club, Star Wars, etc.
   - Call `generate_parody()` for each

2. **Structured Data Extraction** (`test_popular_movies.py`)
   - Capture thinking trace from `<think>` tags
   - Extract tool calls and their results
   - Extract final parody answer
   - Store with template metadata

3. **Dataset File Generation**
   - Create JSONL files:
     - `rlvr_dataset_TIMESTAMP.jsonl`: Full structured data
     - `tool_calls_TIMESTAMP.jsonl`: Tool calls only
     - `reasoning_traces_TIMESTAMP.jsonl`: Reasoning only
   - Summary CSV: `rlvr_summary_TIMESTAMP.csv`

4. **Dataset Conversion** (`rlvr_dataset_tools.py`)
   - Auto-label based on quality criteria (phonetic score ≥ 0.6, humor ≥ 6, 2+ tool calls)
   - Convert to training formats:
     - **SFT**: instruction, output with reasoning and solution
     - **DPO**: prompt, chosen vs rejected responses
     - **RLVR**: Full structured with verifiable rewards

## State Management

**Agent State:**
- Maintained internally by smolagents CodeAgent
- LLM maintains conversation history within single `agent.run()` call
- Tool calls and results passed through agent loop

**File-Based State:**
- Google Drive processor: `~/.parody_processor/processed_state.json`
  - Tracks `{file_id: timestamp}` of processed files
  - Prevents reprocessing same input files

**Output State:**
- Local file system (`./output/`, `./parody_output/`)
  - Sequential numbering: `{number}_{timestamp}_{model}.csv`
  - Per-step debug files for troubleshooting

## Key Abstractions

**CerebrasModel:**
- Purpose: Adapter bridging Cerebras API to smolagents CodeAgent interface
- Location: `generate_parody.py:CerebrasModel`
- Pattern: Callable class implementing `__call__(messages, stop_sequences, **kwargs)`
- Responsibilities:
  - Message formatting for Cerebras API
  - Template tag preprocessing (removes smolagents placeholders)
  - API call execution
  - Error handling and response wrapping

**CodeAgent + Tools:**
- Purpose: Orchestrate parody generation with structured reasoning
- Tools used:
  - `word_phone_tool`: Verify phonetic similarity (0.0-1.0 score)
  - `parody_suggestions_tool`: Pre-compute candidate replacements (loaded from Hugging Face Hub)
- Pattern: Agent autonomously decides when to call tools and interprets results

**OutputCapture:**
- Purpose: Capture model outputs across agent steps for debugging and DPO training
- Location: `generate_parody.py:OutputCapture`
- Methods:
  - `callback(step_log)`: Called after each agent step
  - `extract_data(text)`: Regex-based extraction of attempts, reasoning
  - `export_to_csv()`: Persist extracted data

**RLVRTemplateTags:**
- Purpose: Configurable reasoning/solution tag wrapper for training
- Location: `test_popular_movies.py`, `rlvr_dataset_tools.py`
- Supports: Multiple templates (default, DeepSeek, custom)

## Entry Points

**CLI Single Title:**
- Location: `generate_parody.py:main()`
- Triggers: `python generate_parody.py --title "Title" --model "qwen-3-32b"`
- Responsibilities: Parse args, validate API key, call `generate_parody()`

**CLI Batch Processing:**
- Location: `batch_generate.py:main()`
- Triggers: `python batch_generate.py --input input.csv --output output.csv`
- Responsibilities: Read titles from CSV, process each, write results

**Automated Google Drive Workflow:**
- Location: `drive_batch_processor.py:main()`
- Triggers: GitHub Actions scheduled (every 6 hours) or manual dispatch
- Responsibilities: Connect to Drive, discover new files, process, upload results

**RLVR Test Suite:**
- Location: `test_popular_movies.py:main()`
- Triggers: `python test_popular_movies.py --limit 3`
- Responsibilities: Run parody generation on test movies, capture training data

**Dataset Tools CLI:**
- Location: `rlvr_dataset_tools.py`
- Subcommands: `label`, `auto-label`, `convert`, `stats`
- Triggers: `python rlvr_dataset_tools.py {subcommand} --input file.jsonl --output out.jsonl`

## Error Handling

**Strategy:** Graceful degradation with comprehensive logging

**Patterns:**

**Missing API Key:**
- Check: `os.environ.get("CEREBRAS_API_KEY")`
- Action: Log error, exit with code 1
- Location: `generate_parody.py:main()`

**Tool Load Failures:**
- Wrapped: `load_tool()` calls use `trust_remote_code=True`
- Fallback: None (will raise exception if Hub is unavailable)
- Mitigation: Tools pinned to specific Hub IDs for consistency

**Invalid Model Response:**
- Catch: Exception in `CerebrasModel.__call__()`
- Action: Log full exception, return `ModelResponse(content=f"Error: {str(e)}")`
- Impact: Agent may retry or fail gracefully

**Output Extraction Failures:**
- Pattern matching in `OutputCapture.extract_data()`
- Fallback: Return empty strings for missing fields
- Result: CSV entries with "Generation failed" or empty reasoning

**File I/O Errors:**
- Google Drive: `ValueError` on missing credentials
- Local files: `FileNotFoundError` on missing input.csv
- Logging: All exceptions logged, execution halted

## Cross-Cutting Concerns

**Logging:**
- Framework: Python `logging` module
- Configuration: Both file (`debug.log`) and stream output
- Pattern: Timestamp + name + level + message

**Validation:**
- Phonetic scores: Threshold > 0.6 for acceptance
- CSV input: Must have "title" column
- Suggestions: Fallback to empty list if no matches

**Authentication:**
- Cerebras: Environment variable `CEREBRAS_API_KEY`
- Google Drive: JSON credentials from `GOOGLE_DRIVE_CREDENTIALS` env var

**Configuration:**
- Model selection: CLI arg `--model` (default: "qwen-3-32b")
- Output location: CLI arg `--output-dir` (default: "./output")
- Batch size: No limit (processes all titles in CSV)

---

*Architecture analysis: 2026-01-31*
