# Technology Stack

**Analysis Date:** 2026-01-31

## Languages

**Primary:**
- Python 3.10+ - All application logic, AI agent orchestration, and batch processing

## Runtime

**Environment:**
- Python 3.10+ (specified in GitHub Actions workflow at `parodies2026/.github/workflows/generate-parody.yml`)
- Currently tested with Python 3.13.9
- Requires pip for dependency management

**Package Manager:**
- pip - Primary dependency manager
- Lockfile: `requirements.txt` (no version-locked lockfile like poetry.lock)

## Frameworks

**Core:**
- smolagents >= 1.0.0 - Hugging Face's lightweight agent framework for tool orchestration and code generation
  - Location: Used in `parodies2026/generate_parody.py`
  - Purpose: Enables LLM to generate and execute Python code to call custom tools

**API & LLM:**
- cerebras-cloud-sdk - Cerebras Cloud API client for LLM inference
  - Location: `parodies2026/generate_parody.py` (imported as `from cerebras.cloud.sdk import Cerebras`)
  - Model: Qwen 3-32B (default, configurable via `--model` flag)
  - Purpose: Powers the parody generation agent with fast inference

**Google Integration:**
- google-api-python-client - Google Drive and Google Cloud API client
  - Location: `parodies2026/upload_to_drive.py`, `parodies2026/drive_batch_processor.py`
  - Purpose: Manages input/output workflow with Google Drive folders
- google-auth - OAuth 2.0 and Service Account authentication
  - Location: Used alongside google-api-python-client
- google-auth-oauthlib - Service account and OAuth flow support
- google-auth-httplib2 - HTTP library adapter for Google Auth

**CLI & Formatting:**
- rich - Terminal formatting and colored output
  - Location: `parodies2026/generate_parody.py` (used for rich print formatting)
  - Purpose: Beautiful console output for user feedback

**Phonetic Processing:**
- pronouncing - Python wrapper for CMU Pronouncing Dictionary
  - Purpose: Word pronunciation lookup (used indirectly via smolagents tools)

## Key Dependencies

**Critical:**
- smolagents - Enables the agent to use tools during generation. No fallback implementation exists.
- cerebras-cloud-sdk - All LLM inference depends on this. Must have valid API key.
- google-api-python-client - Required for Google Drive data pipeline. CI/CD depends on this.

**Infrastructure:**
- google-auth, google-auth-oauthlib, google-auth-httplib2 - Service account credential chain for Drive access
- rich - Terminal output only; app functions without it but output less user-friendly
- pronouncing - Dependency of custom Hugging Face tools (patruff/parody-suggestions, patruff/word-phone)

## Configuration

**Environment:**
- `CEREBRAS_API_KEY` - Required. Cerebras API authentication key
  - Set via: Environment variable (GitHub Actions secrets or local export)
  - Used in: `parodies2026/generate_parody.py` line 421, GitHub Actions workflow
- `GOOGLE_DRIVE_CREDENTIALS` - Required for Drive integration. Service account JSON credentials
  - Set via: Environment variable (GitHub Actions secrets)
  - Format: Full JSON content of Google Cloud service account key file
  - Used in: `parodies2026/drive_batch_processor.py`, `parodies2026/upload_to_drive.py`
- `CEREBRAS_MODEL` - Optional. LLM model selection
  - Default: "qwen-3-32b"
  - Configurable via: Command-line flag `--model` or env variable
  - Used in: `parodies2026/generate_parody.py` for model selection

**Build:**
- GitHub Actions workflow: `.github/workflows/generate-parody.yml`
  - Runs on Ubuntu-latest
  - Scheduled: Every 6 hours (cron: `0 */6 * * *`)
  - Manual trigger: Via workflow_dispatch with optional model selection
  - Python setup: 3.10
  - Installs dependencies: `pip install -r requirements.txt`

## Platform Requirements

**Development:**
- Python 3.10+
- pip
- Git (for repository access)
- Internet connection (for Cerebras API, Google Drive, Hugging Face Hub)

**Production (CI/CD):**
- GitHub Actions (runs on ubuntu-latest)
- GitHub repository with Secrets configured:
  - CEREBRAS_API_KEY
  - GOOGLE_DRIVE_CREDENTIALS
- Valid Cerebras Cloud account and API key
- Valid Google Cloud project with:
  - Service Account created and authorized for Drive access
  - Drive API enabled
  - Service account JSON key file

## External Tool Registry

**Hugging Face Hub Tools:**
- `patruff/parody-suggestions` - Loaded via smolagents `load_tool()`
  - Purpose: Finds phonetically similar funny words for a given word
  - Input: target word, word list, min similarity threshold, custom pronunciations
  - Output: List of candidate replacements with similarity scores
- `patruff/word-phone` - Loaded via smolagents `load_tool()`
  - Purpose: Verifies phonetic similarity between two words
  - Input: Two words to compare
  - Output: Similarity score (0.0-1.0, threshold > 0.6 acceptable)

## Data Files

**Static Data:**
- `parodies2026/known100.csv` - Training/reference data
  - Contains: 100 verified funny parody examples with reasoning
  - Format: CSV with columns: original, parody, reasoning
  - Used in: System prompt as few-shot examples

**Input/Output:**
- **Input**: CSV files uploaded to Google Drive `parodiesdata/input/` folder
  - Format: Must have `title` column
  - Processed by: `parodies2026/drive_batch_processor.py` and `parodies2026/batch_generate.py`
- **Output**: CSV results written to Google Drive `parodiesdata/output/` folder
  - Format: Columns include input, parody_result, reasoning, timestamp
  - Filename pattern: `output_[original_filename]_[timestamp].csv`
- **Local Cache**: State tracking file at `~/.parody_processor/processed_state.json`
  - Prevents reprocessing of files

---

*Stack analysis: 2026-01-31*
