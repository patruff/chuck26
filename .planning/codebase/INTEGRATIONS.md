# External Integrations

**Analysis Date:** 2026-01-31

## APIs & External Services

**Cerebras Cloud AI:**
- Service: Cerebras Cloud API for LLM inference
  - SDK/Client: `cerebras-cloud-sdk`
  - Auth: `CEREBRAS_API_KEY` environment variable
  - Model: Qwen 3-32B (default, configurable)
  - Used in: `parodies2026/generate_parody.py` (lines 15-16, 49-57)
  - Endpoint: Cerebras Cloud API (called via SDK)
  - Purpose: Executes the parody generation agent with structured reasoning and tool use

**Hugging Face Hub:**
- Service: Hugging Face Model Hub for tool loading
  - SDK/Client: `smolagents` (includes Hub integration)
  - Auth: No authentication required for public tools
  - Tools loaded:
    - `patruff/parody-suggestions` - Generates phonetically similar word candidates
    - `patruff/word-phone` - Calculates phonetic similarity scores
  - Used in: `parodies2026/generate_parody.py` (lines 42-43)
  - Purpose: Remote tools accessed via smolagents `load_tool()` for agent execution

## Data Storage

**Databases:**
- None detected. Application is stateless for parody generation.

**File Storage:**
- Google Drive (primary data source and sink)
  - Service: Google Drive API v3
  - Client: `google-api-python-client` + `google-auth` + `google-auth-oauthlib`
  - Connection: Service account authentication via `GOOGLE_DRIVE_CREDENTIALS` env var
  - Folder structure in Drive:
    - `parodiesdata/input/` - Source CSV files with titles to process
    - `parodiesdata/output/` - Generated parody results with reasoning
    - `parodiesdata/dpo/` - Human-annotated chosen vs rejected parodies
  - Used in: `parodies2026/drive_batch_processor.py`, `parodies2026/upload_to_drive.py`
  - Auto-creates folder structure if not present

- Local filesystem (development/fallback)
  - Input: `input.csv` in working directory
  - Output: `output.csv` in working directory (when using `batch_generate.py`)
  - Logs: `debug.log` in working directory
  - Debug: `output/`, `batch_output/`, `parody_output/` directories (gitignored)
  - State: `~/.parody_processor/processed_state.json` (tracks processed files)

**Caching:**
- State file at `~/.parody_processor/processed_state.json`
  - Purpose: Prevents reprocessing of files across CI/CD runs
  - Format: JSON tracking processed file IDs and timestamps
  - Used in: `parodies2026/drive_batch_processor.py`

## Authentication & Identity

**Auth Provider:**
- Custom (Service Account based)
  - Implementation approach:
    - **Cerebras**: Direct API key authentication
      - Key format: Plain text string
      - Set via: `CEREBRAS_API_KEY` environment variable
      - Location: `parodies2026/generate_parody.py` line 421
    - **Google Drive**: OAuth 2.0 Service Account
      - Credentials format: JSON service account key file
      - Set via: `GOOGLE_DRIVE_CREDENTIALS` environment variable (full JSON content)
      - Parsed in: `parodies2026/drive_batch_processor.py` lines 44-52, `parodies2026/upload_to_drive.py` lines 34-43
      - Scopes: `['https://www.googleapis.com/auth/drive']` (full Drive access)

**GitHub Secrets (for CI/CD):**
- `CEREBRAS_API_KEY` - Cerebras authentication
- `GOOGLE_DRIVE_CREDENTIALS` - Service account JSON key
- Both injected as environment variables in GitHub Actions workflow

## Monitoring & Observability

**Error Tracking:**
- None detected. No external error tracking service (Sentry, etc.) integrated.

**Logs:**
- Local file logging
  - Handler: `logging.FileHandler("debug.log")`
  - Format: `'%(asctime)s - %(name)s - %(levelname)s - %(message)s'`
  - Used in: `parodies2026/generate_parody.py` lines 31-39
  - Also streams to console via `logging.StreamHandler()`
  - Log files are gitignored and not persisted

## CI/CD & Deployment

**Hosting:**
- GitHub Actions (compute environment)
  - Runtime: `ubuntu-latest`
  - Execution context: GitHub-hosted runner

**CI Pipeline:**
- GitHub Actions workflow: `.github/workflows/generate-parody.yml`
  - **Triggers:**
    - Scheduled: Every 6 hours (cron `0 */6 * * *`)
    - Manual: Via `workflow_dispatch` with optional `model` input parameter
  - **Steps:**
    1. Checkout repository (actions/checkout@v4)
    2. Setup Python 3.10 (actions/setup-python@v5)
    3. Install dependencies (`pip install -r requirements.txt`)
    4. Run processor: `python drive_batch_processor.py`
    5. Generate summary (always runs, displays folder structure)
  - **Environment variables injected:**
    - `CEREBRAS_API_KEY` from secrets
    - `GOOGLE_DRIVE_CREDENTIALS` from secrets
    - `CEREBRAS_MODEL` from workflow input (defaults to "qwen-3-32b")

## Environment Configuration

**Required env vars:**
- `CEREBRAS_API_KEY` - Cerebras Cloud API key (required for all parody generation)
- `GOOGLE_DRIVE_CREDENTIALS` - Google service account JSON (required for Drive integration)

**Optional env vars:**
- `CEREBRAS_MODEL` - LLM model selection (default: "qwen-3-32b")

**Secrets location:**
- GitHub Actions: Settings → Secrets and variables → Actions
- Local development: Export as environment variables before running scripts
  ```bash
  export CEREBRAS_API_KEY="your-key"
  export GOOGLE_DRIVE_CREDENTIALS='{...full json...}'
  ```

## Webhooks & Callbacks

**Incoming:**
- None detected. No webhook endpoints in the application.

**Outgoing:**
- None detected. Application does not call external webhooks.
- However: Generates data to Google Drive asynchronously via Drive API (not a webhook, but async push)

## Data Flow Summary

**Parody Generation Pipeline:**

```
GitHub Actions (scheduled/manual)
    ↓
drive_batch_processor.py
    ↓ (reads)
Google Drive (parodiesdata/input/)
    ↓
generate_parody.py
    ↓ (creates agent)
CerebrasModel + smolagents CodeAgent
    ↓ (loads tools)
Hugging Face Hub (patruff/parody-suggestions, patruff/word-phone)
    ↓ (calls LLM)
Cerebras Cloud API (qwen-3-32b)
    ↓ (generates reasoning + tool calls)
Agent execution (phonetic verification)
    ↓
parse results
    ↓ (writes)
Google Drive (parodiesdata/output/)
```

**Batch Local Pipeline (development):**

```
batch_generate.py
    ↓ (reads)
input.csv
    ↓
generate_parody.py
    ↓ (parody generation same as above)
CerebrasModel + Agent + Cerebras API
    ↓
parse results
    ↓ (writes)
output.csv (local)
```

## Rate Limiting & Quotas

**Cerebras Cloud:**
- No explicit rate limiting mentioned in code
- Assumes standard Cerebras Cloud API rate limits apply
- Model used: Qwen 3-32B (fast inference, suitable for 6-hour batch cycles)

**Google Drive:**
- No quota handling detected
- Relies on standard Google Drive API quotas (varies by account type)
- No retry logic for quota exceeded errors observed

---

*Integration audit: 2026-01-31*
