# Testing Patterns

**Analysis Date:** 2026-01-31

## Test Framework

**Runner:**
- No automated test framework configured (no pytest, unittest, or other framework)
- Manual test script: `test_popular_movies.py` serves as comprehensive test suite
- Test runs via direct script execution: `python test_popular_movies.py [options]`

**Assertion Library:**
- No assertion library - testing relies on manual verification and output inspection

**Run Commands:**
```bash
python test_popular_movies.py                    # Run all 10 movies
python test_popular_movies.py --limit 3          # Test first 3 movies
python test_popular_movies.py --output-dir DIR   # Custom output directory
python test_popular_movies.py --template deepseek # Use deepseek template
python test_popular_movies.py --show-templates    # Show available templates
python test_popular_movies.py --list-movies       # List 10 test movies
```

## Test File Organization

**Location:**
- `test_popular_movies.py` - Main test file at project root `/Users/patruff/chucklesPRIME/parodies2026/`
- Tests located alongside source code (co-located pattern)
- Test data defined within test file: `POPULAR_MOVIES = [...]`
- No separate test directory

**Naming:**
- Test file: `test_popular_movies.py` (follows test_* convention)
- Test functions: `run_popular_movies_test()`, `generate_parody_with_capture()`, `save_rlvr_dataset()`
- Helper classes: `RLVRTemplateTags`, `ToolCall`, `ParodyAttempt`, `RLVRDataPoint`, `EnhancedOutputCapture`

**Structure:**
```
test_popular_movies.py
├── RLVR Template Configuration (@dataclass RLVRTemplateTags)
├── Template Presets (DEFAULT_TEMPLATE, DEEPSEEK_TEMPLATE)
├── Test Data (POPULAR_MOVIES list)
├── Data Point Classes (@dataclass ToolCall, ParodyAttempt, RLVRDataPoint)
├── Capture Class (EnhancedOutputCapture)
├── Processing Functions (generate_parody_with_capture, save_rlvr_dataset)
├── Main Test Runner (run_popular_movies_test)
└── CLI Entry Point (main, argparse)
```

## Test Structure

**Test Suite Organization:**
- Test is a complete pipeline: input titles → generate parodies → capture outputs → extract structured data → save results
- No individual unit test assertions
- Validation through successful completion and output generation

**Test Flow:**
1. Parse CLI arguments (limit, model, output directory, template)
2. Load test movies (10 popular movies)
3. For each movie:
   - Call `generate_parody_with_capture()`
   - Capture and structure output
   - Extract thinking trace, tool calls, final parody
   - Calculate quality signals
4. Save results in 7 different formats (JSONL, JSON, CSV, etc.)
5. Log summary statistics

**Example test structure from `test_popular_movies.py`:**
```python
def run_popular_movies_test(
    limit: Optional[int] = None,
    model_name: str = "qwen-3-32b",
    api_key: Optional[str] = None,
    output_dir: str = "./rlvr_output",
    template: RLVRTemplateTags = None
):
    """
    Run the popular movies test and generate RLVR training data.
    """
    template = template or DEFAULT_TEMPLATE
    movies = POPULAR_MOVIES[:limit] if limit else POPULAR_MOVIES

    data_points: List[RLVRDataPoint] = []

    for i, movie in enumerate(movies, 1):
        logging.info(f"\n{'='*60}")
        logging.info(f"[{i}/{len(movies)}] Processing: {movie}")
        logging.info(f"{'='*60}")

        try:
            data_point = generate_parody_with_capture(...)
            data_points.append(data_point)
            logging.info(f"  Output: {data_point.final_parody}")
        except Exception as e:
            logging.error(f"Error processing '{movie}': {e}")
            error_point = RLVRDataPoint(
                input_title=movie,
                final_parody=f"ERROR: {str(e)}",
                quality_label="error"
            )
            data_points.append(error_point)

    saved_files = save_rlvr_dataset(data_points, ...)
    return data_points
```

## Mocking

**Framework:** No mocking framework configured

**Patterns:**
- No mocks used - test calls real `generate_parody()` function
- Real API calls made to Cerebras (requires `CEREBRAS_API_KEY` environment variable)
- Real file I/O to disk
- Real tool executions (word_phone_tool, parody suggestions)

**What to Mock (if unit testing were added):**
- Cerebras API calls: `cerebras.cloud.sdk.Cerebras` client
- File I/O operations for isolated testing
- External tools: `word_phone_tool`, `parody_suggestions_tool`

**What NOT to Mock:**
- Data extraction/parsing logic (regex patterns should be tested against real output)
- Dataclass construction and serialization
- Configuration template formatting

## Fixtures and Factories

**Test Data:**
- Constant list of 10 well-known movies: `POPULAR_MOVIES = ["The Matrix", "Die Hard", ...]`
- CSV files used for data: `known100.csv`, `input.csv`, `test.csv`
- Known parodies loaded from CSV: `load_known_parodies()` from `word_structures.py`

**Data Point Factories:**
```python
# RLVRDataPoint created during test
data_point = RLVRDataPoint(
    input_title=title,
    model_name=model_name,
    template_name="custom" if template else "default"
)

# ToolCall created during output parsing
tool_calls.append(ToolCall(
    tool_name="word_phone_tool",
    arguments={"word1": word1, "word2": word2},
    result=score_float
))

# Error data points for failed generations
error_point = RLVRDataPoint(
    input_title=movie,
    final_parody=f"ERROR: {str(e)}",
    quality_label="error"
)
```

**Location:**
- Fixtures defined in `test_popular_movies.py`
- Known parodies loaded from `known100.csv` via `word_structures.py`
- Example prompts from `system_prompt.py`

## Coverage

**Requirements:** No coverage tracking configured

**Coverage Approach:**
- Full integration test covering entire pipeline
- Tests real-world scenario: batch processing of movie titles
- Output inspection is manual (check CSV files, JSONL datasets, logs)

**View Coverage:**
- Not applicable - no coverage tool configured
- Validation through output files generated:
  - `rlvr_dataset_{timestamp}.jsonl` - Full data points
  - `rlvr_summary_{timestamp}.csv` - Quick review
  - `tool_calls_{timestamp}.jsonl` - Tool usage extraction
  - `reasoning_traces_{timestamp}.jsonl` - Thinking traces
  - `rlvr_training_{timestamp}.jsonl` - RLVR format

## Test Types

**Integration Tests:**
- `test_popular_movies.py` is a comprehensive integration test
- Tests full pipeline: generation → capture → extraction → storage
- Scope: End-to-end movie title parody generation with RLVR data collection
- Approach: Run complete workflow, capture outputs, extract structured data, validate quality signals

**Batch Processing Tests:**
- `batch_generate.py` processes multiple titles from CSV
- Manual testing by running: `python batch_generate.py --input input.csv --output output.csv`
- Validates error handling for each title independently
- Records both successes and failures

**Unit Tests:**
- Not present in codebase
- No automated unit test framework
- Individual components tested via integration test outputs

**E2E Tests:**
- `test_popular_movies.py` serves as E2E test
- Tests against real Cerebras API (requires valid credentials)
- Produces RLVR training datasets as evidence of success
- Cannot be run without external API access

## Common Patterns

**Data Extraction Testing:**
- Regex patterns tested implicitly through successful parsing
- Multiple patterns tested for robustness: primary pattern + fallback pattern
- Example from `test_popular_movies.py`:
```python
# Pattern to match word_phone_tool calls
tool_pattern = r'word_phone_tool\s*\(\s*["\']([^"\']+)["\']\s*,\s*["\']([^"\']+)["\']\s*\).*?(?:->|:|\=)\s*([\d.]+)'

# Alternative pattern for different formats
alt_pattern = r'[Oo]riginal\s*["\']?(\w+)["\']?\s*(?:vs|→|->)\s*["\']?(\w+)["\']?\s*:\s*([\d.]+)'

# Try primary, then fallback
matches = re.finditer(tool_pattern, text, re.IGNORECASE | re.DOTALL)
for match in matches:
    # ... process matches

# Also try alternative pattern
alt_matches = re.finditer(alt_pattern, text)
for match in alt_matches:
    # ... process with duplicate detection
```

**Quality Signal Testing:**
- Tests calculate and verify quality signals on extracted data:
```python
def calculate_quality_signals(self):
    """Calculate quality signals from the extracted data."""
    all_scores = []

    for tc in self.data_point.tool_calls:
        if tc.tool_name == "word_phone_tool" and isinstance(tc.result, (int, float)):
            all_scores.append(float(tc.result))

    if all_scores:
        self.data_point.average_phonetic_score = sum(all_scores) / len(all_scores)
        self.data_point.all_phonetic_scores_valid = all(s > 0.6 for s in all_scores)
```

**Output Format Testing:**
- Tests save outputs in multiple formats for validation:
  1. JSONL (line-delimited JSON) - streaming
  2. JSON (full JSON) - inspection
  3. CSV (simplified summary) - quick review
  4. Tool calls separately - tool-use training
  5. Reasoning traces - reasoning training
  6. RLVR training format - ready for fine-tuning
  7. Template configuration - reference docs

**Template Testing:**
- Tests with multiple preset templates: default, deepseek, or custom
- Example:
```python
# Load preset template
template = TEMPLATES.get(args.template, DEFAULT_TEMPLATE)

# Or create custom from CLI args
template = RLVRTemplateTags(
    reasoning_start=args.reasoning_start or base_template.reasoning_start,
    reasoning_end=args.reasoning_end or base_template.reasoning_end,
    solution_start=args.solution_start or base_template.solution_start,
    solution_end=args.solution_end or base_template.solution_end,
)

# Verify format through output
entry = dp.to_rlvr_training_format(template)
```

**Error Recovery Testing:**
- Tests handle individual movie failures without stopping entire batch
- Failed items tracked with `quality_label="error"`
- Error messages preserved in final parody field
- Example:
```python
try:
    data_point = generate_parody_with_capture(...)
except Exception as e:
    logging.error(f"Error processing '{movie}': {e}")
    error_point = RLVRDataPoint(
        input_title=movie,
        model_name=model_name,
        final_parody=f"ERROR: {str(e)}",
        quality_label="error"
    )
    data_points.append(error_point)  # Continue processing
```

## Manual Testing Procedures

**Running the full test suite:**
```bash
python test_popular_movies.py --limit 2 --output-dir ./test_output
```

**Validating outputs:**
1. Check `./test_output/datasets/rlvr_summary_*.csv` for quick results
2. Inspect `./test_output/datasets/rlvr_dataset_*.json` for full data
3. Review logs: `debug.log` for API calls and errors

**Testing specific functionality:**
- Tool extraction: check `./test_output/datasets/tool_calls_*.jsonl`
- Reasoning quality: check `./test_output/datasets/reasoning_traces_*.jsonl`
- Training format: check `./test_output/datasets/rlvr_training_*.jsonl`

---

*Testing analysis: 2026-01-31*
