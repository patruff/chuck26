# Pitfalls Research

> Research findings for chucklesPRIME: Common pitfalls when building RLVR training data
> generation pipelines with smolagents and HuggingFace tooling.
>
> Date: 2026-01-31
> Sources: arxiv papers (2025-2026), HuggingFace docs, smolagents GitHub issues,
> TRL/GRPO documentation, CMU Pronouncing Dictionary research, and analysis of
> the existing parodies2026/ codebase.

---

## Critical Pitfalls

These are the issues most likely to cause project failure if not addressed early.

### CP-1: Reward Hacking via Phonetic Score Gaming

**What happens:** The model learns to exploit the phonetic scoring function rather than
produce genuinely funny parodies. For example, it may pick words that score high on
rhyme_score but are semantically meaningless, or repeat known-good substitutions
without creativity.

**Warning signs:**
- Generated parodies all use the same small set of replacement words
- High phonetic scores but low human humor ratings
- Model outputs the "answer" immediately in `<think>` tags without reasoning
- Parodies converge on a narrow set of patterns (e.g., always food-based substitutions)

**Prevention:**
- Use composite reward functions: phonetic score (verifiable) + humor diversity penalty
- Add negative rewards for repeating substitutions seen in training data
- Track the distribution of replacement words across generations; flag if entropy drops
- Consider RLVRR-style decomposed rewards: phonetic correctness (deterministic) + humor style (LLM-verified)

**Phase:** Architecture (reward function design), Data Generation (diversity monitoring)

**Existing codebase issue:** The current `auto_label_data_point()` in `rlvr_dataset_tools.py`
uses a single `min_phonetic_score` threshold (0.6) as the primary quality gate. This
binary pass/fail creates a trivially gameable reward signal.

---

### CP-2: Diversity Collapse in Training Data

**What happens:** RLVR training with GRPO concentrates probability mass on a narrow
set of high-reward outputs. The model becomes faster at producing ONE type of parody
but loses the ability to generate diverse, creative alternatives.

**Warning signs:**
- Pass@1 improves but Pass@k degrades (model succeeds more often but with less variety)
- Entropy of generated outputs decreases over training steps
- Out-of-distribution titles produce gibberish or repetitive outputs
- All parodies start using the same humor pattern (e.g., always crude, never absurd)

**Prevention:**
- Use forward-KL or JS-divergence instead of reverse-KL in the training objective
  (see DPH-RL framework from arxiv:2509.07430)
- Implement difficulty-based curriculum: start with easy titles, progressively add harder ones
- Monitor Pass@k metrics alongside Pass@1 during training
- Use training problem augmentation: generate variations of input titles
- Apply Differential Smoothing to preserve diversity in correct trajectories

**Phase:** Data Generation (diversity in training prompts), Training (objective function choice)

**Research citation:** "Diversity collapse manifests as reduced entropy, diminished
Pass@k performance, catastrophic forgetting of alternative strategies, and diminished
robustness to environment variation." (arxiv:2509.07430)

---

### CP-3: smolagents Code Parsing Failures with Non-OpenAI Models

**What happens:** CodeAgent expects LLM output wrapped in specific markdown code block
format. Models like Qwen3 via Cerebras frequently produce output that fails the
regex-based parser, causing SyntaxError loops or infinite retries.

**Warning signs:**
- Repeated "Error in code parsing: Your code snippet is invalid" messages
- Agent reaches max_steps without producing output
- Partial stop sequences appearing in parsed code (e.g., `</code` without `>`)
- Streaming mode causes infinite syntax error cycles

**Prevention:**
- Pin smolagents version and test thoroughly with each LLM backend
- Set `stream_outputs=False` (smolagents issue #1872 confirms streaming breaks CodeAgent)
- Implement output sanitization in the model wrapper (strip partial stop sequences)
- Add post-processing to clean code blocks before they reach the parser
- Set reasonable `max_steps` (10-15) with graceful fallback on exhaustion
- Consider ToolCallingAgent as alternative for models that struggle with code output format

**Phase:** Architecture (model wrapper design), Testing (cross-backend validation)

**Existing codebase issue:** The `CerebrasModel` class in `generate_parody.py` does basic
message preprocessing but does NOT sanitize LLM output for stop sequence artifacts.
The `_preprocess_content()` method only handles input templates, not output parsing.
Error handling returns `ModelResponse(content=f"Error: {str(e)}")` which will itself
fail code parsing.

---

### CP-4: Verifier Imperfection (False Positives and False Negatives)

**What happens:** The phonetic scoring function (the "verifier" in RLVR terms) is
systematically wrong in both directions: it rejects good parodies (false negatives)
and accepts bad ones (false positives).

**Warning signs:**
- Human reviewers disagree with auto-labels more than 20% of the time
- Known-funny parodies from `known100.csv` score below the 0.6 threshold
- Boring/obvious parodies pass all phonetic checks
- Words not in CMU dictionary silently fail or get incorrect scores

**Prevention:**
- Audit the verifier against known-good examples: run `word_phone_tool` on all
  100 known parodies and check which ones would fail auto-labeling
- Track false positive and false negative rates with human-reviewed holdout set
- Apply noise-corrected RLVR (backward or forward correction) to handle verifier noise
- Expand custom_phones dictionary for words the model frequently tries to use

**Phase:** Data Generation (verifier calibration), Architecture (noise correction)

**Research citation:** "A recent analysis of a math-RL dataset found that over 38%
of responses flagged as incorrect by a rule-based system were in fact correct."
(arxiv:2510.00915)

---

### CP-5: RLVR Trains Speed, Not New Capability

**What happens:** RLVR primarily performs "search compression" -- it teaches the model
to find known-good answers faster, but does NOT expand the model's actual capability
frontier. If the base Qwen3 model cannot produce a certain type of parody in k
attempts, RLVR training will not teach it that capability.

**Warning signs:**
- Trained model produces outputs identical to base model's best-of-N outputs
- No improvement on titles the base model consistently fails on
- Performance gains disappear on contamination-free evaluation sets

**Prevention:**
- Before training, establish a baseline: run base model N times per title, track Pass@k
- Focus training data on "sweet spot" difficulty: titles where base model succeeds
  30-70% of the time (not always, not never)
- For capabilities the base model lacks, use SFT first to seed the capability,
  then RLVR to sharpen it
- Validate on held-out, distribution-shifted test sets (different title genres,
  different humor styles)

**Phase:** Data Generation (difficulty filtering), Evaluation (capability assessment)

**Research citation:** "If your model can solve a problem in 8 tries, RLVR trains it
to succeed in 1 try. This is primarily search compression, not expanded reasoning
capability." (promptfoo.dev/blog/rlvr-explained)

---

## RLVR-Specific Pitfalls

### RL-1: Training Data Contamination Creates Illusory Gains

**What happens:** If the base model was pre-trained on data that includes known parodies
(e.g., the same movie title puns that appear in known100.csv), RLVR training on those
titles will show inflated improvement that does not generalize.

**Warning signs:**
- Model produces correct parodies for known100.csv titles on first attempt without tool use
- Performance on known titles vastly exceeds performance on novel titles
- Qwen3 models show suspicious sensitivity to specific prompt formats

**Prevention:**
- Split evaluation into "potentially contaminated" (famous titles) and "definitely clean"
  (obscure/synthetic titles) sets
- Include synthetic/invented titles in evaluation (e.g., made-up movie names)
- Track tool usage rate: if model skips tools on familiar titles, contamination is likely
- Do not use all known100.csv examples for both training and evaluation

**Phase:** Data Generation (train/eval split), Evaluation (contamination analysis)

---

### RL-2: Reward Function Granularity Mismatch

**What happens:** Binary rewards (0/1 for phonetic threshold) provide insufficient
signal for GRPO to learn meaningful distinctions. A parody scoring 0.61 and one
scoring 0.99 both get reward=1, destroying gradient information.

**Warning signs:**
- Training loss plateaus early
- Model produces "barely passing" outputs (scores cluster just above threshold)
- No qualitative improvement despite continued training

**Prevention:**
- Use continuous reward scores, not binary thresholds
- Decompose reward into multiple components: phonetic_score * humor_bonus * format_compliance
- Weight different phonetic dimensions differently (rhyme_score most important)
- Consider composite rewards that add penalties for shortcut behaviors
  (per arxiv:2509.15557)

**Phase:** Architecture (reward function design)

**Existing codebase issue:** `rlvr_dataset_tools.py` stores `phonetic_validity` as
boolean and `average_phonetic_score` as float, but the auto-labeler uses only the
boolean. The rich score information is captured but then collapsed to binary.

---

### RL-3: GRPO Dataset Format Mismatch

**What happens:** TRL's GRPOTrainer expects a specific dataset format with a `prompt`
column (string or list of chat messages). The current codebase outputs CSV/JSONL
with different column names (`input`, `instruction`, etc.), requiring non-obvious
transformation.

**Warning signs:**
- GRPOTrainer crashes on load with column-not-found errors
- Reward functions receive unexpected kwargs
- Chat template mismatch between training format and model's expected input format

**Prevention:**
- Standardize on TRL's expected format from the start:
  ```python
  {"prompt": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
   "answer": "ground_truth_parody"}
  ```
- Map column names during conversion (rename `input_title` to `prompt`, etc.)
- Test the full pipeline end-to-end with a tiny dataset before generating large volumes
- Include `answer` column with the ground-truth parody for reward verification

**Phase:** Architecture (dataset schema), Data Generation (output format)

---

### RL-4: Entropy Collapse During GRPO Training

**What happens:** As GRPO training progresses, model output entropy collapses. The
model becomes deterministic, always producing the same output for a given input.
In-distribution accuracy rises but out-of-distribution performance degrades.

**Warning signs:**
- Training reward curves are smooth and steadily increasing (too good to be true)
- Model always produces identical output for the same title (zero stochasticity)
- Generated parodies become formulaic (same humor pattern every time)

**Prevention:**
- Monitor entropy at each training checkpoint
- Use temperature > 0 during generation rollouts in GRPO
- Implement curriculum learning: progressively introduce harder titles
- Consider using median instead of mean for GRPO group baseline (more robust to outliers)
- Apply curiosity-driven exploration or Lookahead Tree-Based Rollouts for diversity

**Phase:** Training (hyperparameters, monitoring)

---

## smolagents Pitfalls

### SA-1: Infinite Agent Loops from Code Parse Failures

**What happens:** When the LLM produces malformed code blocks, smolagents enters a
retry loop. Each retry produces a similar malformed output, consuming all max_steps
without producing useful results.

**Warning signs:**
- Agent log shows repeated "Error in code parsing" messages
- Each step produces the same error
- Step callbacks fire rapidly without meaningful content changes
- Raw output files show truncated or garbled code blocks

**Prevention:**
- Set `max_steps` conservatively (10-15 for parody generation)
- Implement step_callback that detects repeated errors and aborts early
- Add output sanitization in model wrapper to fix common formatting issues
- Log and count parsing errors; abort after 3 consecutive parse failures
- Test with multiple LLM backends to find which produce cleanest code blocks

**Phase:** Architecture (error handling), Testing (cross-model validation)

**Existing codebase issue:** `generate_parody.py` creates the CodeAgent but sets no
explicit `max_steps`. The default smolagents max_steps may be too high for this use
case, causing long waits on parse failure loops.

---

### SA-2: Tool Import Authorization Errors

**What happens:** CodeAgent's sandboxed Python interpreter blocks imports and function
calls not explicitly authorized. The model tries to import libraries or call functions
that are not in the allowed list, causing InterpreterError.

**Warning signs:**
- Error: "It is not permitted to evaluate other functions than the provided tools"
- Error: "Code execution failed due to an unauthorized import"
- Agent generates correct logic but uses unauthorized Python stdlib functions

**Prevention:**
- Set `additional_authorized_imports` comprehensively:
  `["json", "re", "string", "math"]` at minimum
- Do NOT include `smolagents` or `load_tool` in authorized imports (tools should be
  pre-loaded, not loaded at runtime inside agent code)
- Test the agent with diverse prompts to discover what imports it tries to use
- Consider pre-computing phonetic suggestions outside the agent loop (as currently
  done) to minimize in-agent tool calls

**Phase:** Architecture (agent configuration)

**Existing codebase issue:** `generate_parody.py` sets
`additional_authorized_imports=["json", "smolagents", "load_tool"]`. Including
`smolagents` and `load_tool` as authorized imports is problematic -- these should be
pre-loaded tools, not runtime imports. This may cause security sandbox issues.

---

### SA-3: smolagents Version Instability

**What happens:** smolagents is an experimental library that breaks between versions.
Model provider integrations (including Cerebras) have been broken by updates.
Code that works on v1.8.1 may fail on v1.9.1+.

**Warning signs:**
- Tests pass locally but fail in CI/CD
- "Model not found" errors after library updates
- Changed API signatures or removed functions
- New required parameters in existing functions

**Prevention:**
- Pin smolagents to a specific version in requirements.txt (e.g., `smolagents==1.23.0`)
- Abstract all smolagents interactions behind an internal interface layer
- Write integration tests that exercise the full agent loop with each supported model
- Monitor smolagents GitHub releases and breaking changes

**Phase:** Architecture (dependency management), Testing (integration tests)

**Existing codebase issue:** `requirements.txt` specifies `smolagents>=1.0.0` which
allows any version. This is dangerously permissive for an experimental library.

---

### SA-4: Output Capture Fragility

**What happens:** The OutputCapture callback relies on specific attribute names in
step_log objects (`llm_output`, `action_output`) and specific text patterns in the
LLM output. Changes to smolagents internals or LLM output format break extraction.

**Warning signs:**
- CSV files contain "Unknown" or empty fields
- Regex patterns in `extract_data()` return no matches
- Thinking traces are empty despite model clearly reasoning
- Step dumps show data exists but callback cannot parse it

**Prevention:**
- Use smolagents' official logging/telemetry features instead of manual attribute inspection
- Make regex patterns more flexible with fallback extractors
- Log raw step_log attributes to identify what fields are actually present
- Test output capture with actual LLM responses, not synthetic test data

**Phase:** Architecture (output capture redesign)

**Existing codebase issue:** `OutputCapture.extract_data()` uses rigid regex patterns
like `r'### Attempt (\d+):\s*\n\*\*"([^"]+)"\*\*'`. These break if the model
outputs attempts in a slightly different format (different heading level, no bold,
no quotes). The same fragile extraction is duplicated in `batch_generate.py`'s
`extract_parody_from_result()`.

---

### SA-5: Tool Loading from Hub is a Network Dependency

**What happens:** `load_tool("patruff/parody-suggestions", trust_remote_code=True)` and
`load_tool("patruff/word-phone", trust_remote_code=True)` download tool code from
HuggingFace Hub at runtime. Network issues, Hub downtime, or repository changes
break the pipeline silently.

**Warning signs:**
- Intermittent failures in batch processing
- Different behavior after updating Hub tools without version pinning
- `trust_remote_code=True` security risk if repository is compromised

**Prevention:**
- Vendor tools locally (copy tool code into the project) for production runs
- Use Hub tools only for distribution, not as a runtime dependency
- Pin to specific Hub revision if using remote loading
- Add health check that validates tool loading before starting batch

**Phase:** Architecture (dependency management)

**Existing codebase issue:** Both `parody_suggestions.py` and `word_phone.py` exist
locally but `generate_parody.py` loads them from the Hub instead. This creates an
unnecessary network dependency when local copies are available.

---

## LLM Backend Pitfalls

### LB-1: Response Format Inconsistency Across Providers

**What happens:** Different LLM providers return responses in different formats.
Cerebras, OpenAI, Anthropic, and local models (Ollama) all have subtly different
message structures, stop sequence behavior, and error formats.

**Warning signs:**
- Code works with one provider but fails with another
- Stop sequences are partially included in output (some providers include, some don't)
- Different providers handle `max_tokens` differently (some count it toward rate limits,
  some don't)
- Model returns empty response or truncated response without error

**Prevention:**
- Build a model adapter interface that normalizes all responses to a common format
- Test with at least 3 different providers in CI
- Handle provider-specific quirks in the adapter, not in business logic
- Normalize: response.content should always be a clean string, never None
- Add retry logic for transient provider errors (rate limits, timeouts)

**Phase:** Architecture (model adapter layer)

**Existing codebase issue:** `CerebrasModel` in `generate_parody.py` is tightly coupled
to the Cerebras SDK. The `_preprocess_content()` method handles smolagents template
tags but does not abstract the provider interface. Switching to a different provider
requires writing a new model class.

---

### LB-2: Token Limit Mismatches

**What happens:** Parody generation prompts can be large (system prompt + examples +
suggestions + style guide). Different models have different context windows. A prompt
that fits in Qwen3-32B's context may exceed a smaller model's limit.

**Warning signs:**
- Truncated outputs that cut off mid-reasoning
- Model produces incomplete parodies (missing verification steps)
- Silent context overflow where model ignores part of the prompt

**Prevention:**
- Calculate prompt token count before sending (use tiktoken or model-specific tokenizer)
- Implement dynamic prompt truncation: reduce examples if prompt is too long
- Make the number of included examples configurable and adaptive
- Set `max_tokens` for response appropriately based on remaining context budget
- Log prompt and response token counts for monitoring

**Phase:** Architecture (prompt management)

**Existing codebase issue:** `build_generation_prompt()` includes ALL example text,
the full style guide, AND all suggestions JSON with no size management. The
`get_example_prompt_text()` defaults to 10 examples but has no token budget awareness.

---

### LB-3: Rate Limiting Varies by Provider

**What happens:** Different providers impose different rate limits (requests per minute,
tokens per minute) and handle violations differently (HTTP 429, queueing, silent drops).
Batch processing hits these limits unpredictably.

**Warning signs:**
- Batch processing works for first N titles then fails
- Intermittent HTTP errors in logs
- Increasing latency as batch progresses
- Cost spikes from retried requests

**Prevention:**
- Implement token-aware rate limiting (not just request-based)
- Add exponential backoff with jitter on 429 responses
- Use a request queue with configurable concurrency per provider
- Cache results: if a title has already been processed, do not reprocess
- Log token usage and cost estimates per request

**Phase:** Architecture (rate limiting), Batch Processing (queue management)

**Existing codebase issue:** `batch_generate.py` processes titles sequentially with
no rate limiting, no backoff, and no caching. If processing fails mid-batch, there
is no resume capability.

---

### LB-4: CerebrasModel Error Handling Returns Invalid Agent Input

**What happens:** When the Cerebras API call fails, `CerebrasModel.__call__()` returns
`ModelResponse(content=f"Error: {str(e)}")`. This error string is then treated as
valid LLM output by the CodeAgent, which tries to parse it as Python code, causing
a cascade of parsing failures.

**Warning signs:**
- Agent enters infinite loop after an API error
- Log shows API error followed by code parsing errors
- Agent wastes all remaining steps trying to parse error messages

**Prevention:**
- Raise an exception on API failure instead of returning error as content
- Implement retry logic within the model wrapper for transient errors
- Use smolagents' built-in error handling mechanisms
- If returning error content, format it as a valid agent message (not code)

**Phase:** Architecture (error handling)

---

## Dataset Pipeline Pitfalls

### DP-1: Missing config_name in HuggingFace Dataset YAML

**What happens:** Uploaded datasets fail to render in the Dataset Viewer because the
YAML metadata lacks `config_name`, which is required even for single-config datasets.

**Warning signs:**
- Dataset uploads successfully but viewer shows "Config names error"
- `load_dataset()` fails with config-related errors
- Dataset appears on Hub but cannot be browsed

**Prevention:**
- Always include `config_name: default` in dataset card YAML
- Use `push_to_hub()` from the datasets library (auto-configures viewer)
- Validate dataset card YAML before upload
- Test `load_dataset()` round-trip after upload

**Phase:** Dataset Pipeline (upload validation)

---

### DP-2: JSONL vs Parquet Format Decision

**What happens:** JSONL is human-readable and easy to generate but performs poorly
for large datasets. Parquet is efficient but harder to debug. Choosing wrong format
causes either performance problems or debugging nightmares.

**Warning signs:**
- Large JSONL files take forever to load
- Dataset viewer is slow or disabled for large datasets
- `to_parquet()` collapses multi-file sharded datasets into single files

**Prevention:**
- Generate in JSONL during development (human-readable, easy to inspect)
- Convert to Parquet for production/training using datasets library
- Keep sharding when converting: use `dataset.to_parquet()` with shard size control
- For datasets < 1GB, JSONL is fine for everything

**Phase:** Dataset Pipeline (format management)

---

### DP-3: Schema Drift Between Generation and Training

**What happens:** The data schema evolves during development (new fields added,
fields renamed, type changes) but previously generated data uses the old schema.
Mixing old and new data causes training failures.

**Warning signs:**
- KeyError when processing older data files
- Inconsistent column presence across dataset splits
- Training crashes on unexpected null values

**Prevention:**
- Define dataset schema in a single source of truth (dataclass or JSON schema)
- Version the schema and include version field in every data point
- Write migration scripts for schema changes
- Validate all data against current schema before training

**Phase:** Architecture (schema definition), Dataset Pipeline (validation)

**Existing codebase issue:** The RLVR output format in `convert_to_rlvr_format()` is
a nested dict with multiple levels (`response_structured`, `rewards`,
`verifiable_checks`). This complex schema is defined implicitly in code, not
declaratively. Any change requires updating multiple functions.

---

### DP-4: CSV Output Fragility

**What happens:** Current codebase outputs results as CSV files. CSV cannot represent
nested structures (tool calls, verification scores), leading to data loss or mangled
output.

**Warning signs:**
- Reasoning text with commas or newlines corrupts CSV rows
- Tool call details are lost or flattened to strings
- Cannot round-trip data through CSV without information loss

**Prevention:**
- Use JSONL as the primary intermediate format (supports nested structures)
- Use CSV only for human review/spreadsheet viewing
- Implement proper JSONL output alongside CSV in the generation pipeline
- Validate that all fields survive serialization/deserialization

**Phase:** Architecture (output format)

**Existing codebase issue:** `OutputCapture.export_to_csv()` outputs a flat CSV with
columns `['input', 'parody1', 'parody2', 'reasoning', 'thinking_trace']`. This loses
tool call details, phonetic scores, and step-by-step verification data that is
critical for RLVR training.

---

### DP-5: No Deduplication or Idempotency

**What happens:** Running batch generation twice on the same input creates duplicate
entries. Merging datasets from multiple runs without deduplication inflates the
dataset with redundant examples, biasing training.

**Warning signs:**
- Dataset has multiple entries for the same input title
- Training data size grows faster than expected
- Model shows bias toward titles that were processed multiple times

**Prevention:**
- Generate unique IDs for each data point (hash of input + timestamp + model)
- Implement deduplication before dataset conversion
- Add idempotency keys: skip titles that already have a result
- Track provenance: which run, which model, which timestamp produced each data point

**Phase:** Data Generation (dedup), Dataset Pipeline (merge logic)

---

## Parody-Specific Pitfalls

### PS-1: CMU Dictionary Out-of-Vocabulary (OOV) Words

**What happens:** Many words the model wants to use for parodies are not in CMU
Pronouncing Dictionary: slang, proper nouns, neologisms, compound words, and
intentional misspellings. The tool returns "Word not found" and the model either
gives up or picks a worse alternative.

**Warning signs:**
- High rate of "Word not found in dictionary" errors in tool call results
- Model avoids creative neologisms and sticks to common words
- Parody quality limited to CMU dictionary vocabulary (~134K words)
- Known funny words like "serbed" require manual custom_phones entries

**Prevention:**
- Expand custom_phones dictionary aggressively (currently only 4 entries)
- Implement grapheme-to-phoneme (g2p) fallback for unknown words
  (use `g2p_en` library or similar)
- Pre-compute and cache pronunciations for all words in the funny_words list
- Log all OOV encounters to identify high-value additions to custom_phones
- Allow the model to propose and justify its own phonetic transcriptions

**Phase:** Architecture (g2p fallback), Data Generation (OOV tracking)

**Existing codebase issue:** `custom_phones` in `word_structures.py` has only 4 entries:
"serbed", "codfather", "foodfellas", "graveheart". This is far too few for creative
parody generation. The funny_words list has ~200 words, but many creative
combinations would produce words not in CMU dictionary.

---

### PS-2: Phonetic Scoring Does Not Capture Human Perception

**What happens:** The current phonetic similarity algorithm uses a weighted combination
of rhyme_score (40-60%), length_score, stress_score, etc. But human perception of
"sounding alike" is more holistic and context-dependent. "Matrix" and "Mattress" are
perceptually very similar but score poorly on formal phonetic metrics.

**Warning signs:**
- Human reviewers rate parodies as sounding similar but tool gives low scores
- Parodies that only differ by one vowel (clearly similar) get mediocre scores
- The model avoids actually-great parodies because the tool rejects them

**Prevention:**
- Calibrate scoring weights against human perceptual judgments
- Run the scoring algorithm on all 100 known parodies and check alignment
- Consider adding a "whole-word perceptual similarity" component that compares
  the full phoneme sequences (not just endings)
- Weight initial consonants more heavily (humans notice word beginnings strongly)
- Add a "semantic surprise" bonus for unexpected but real-word substitutions

**Phase:** Architecture (scoring algorithm refinement)

**Existing codebase issue:** `word_phone.py` and `parody_suggestions.py` have
DIFFERENT similarity calculation algorithms. `word_phone.py` uses rhyme (60%) +
length (30%) + stress (10%). `parody_suggestions.py` uses rhyme (40%) + primary
vowel (15%) + near_rhyme (15%) + special_pattern (15%) + length (5%) + stress (5%)
+ front_consonant (5%). These inconsistent algorithms mean the suggestion tool and
verification tool may disagree on whether a word pair is "similar enough."

---

### PS-3: Funny Words List is Static and Limiting

**What happens:** The funny_words list in word_structures.py contains ~200 hardcoded
words. Parody generation is constrained to these words as replacement candidates,
severely limiting creativity. The model cannot discover that "Schindler's List" ->
"Swindler's Fist" is funny because "swindler" and "fist" are not in the funny words list.

**Warning signs:**
- Generated parodies feel repetitive (same replacement words appearing)
- Parodies for new titles converge on similar patterns
- Model suggests great words that are not in the funny_words list

**Prevention:**
- Make funny_words configurable and loadable from external file
- Add a "dynamic discovery" mode where the model can suggest any word
  and the tool just scores it (current architecture supports this via word_phone_tool)
- Expand the list significantly (1000+ words across more categories)
- Allow per-run or per-title word list customization
- Track which funny_words are actually used vs never used; prune dead words

**Phase:** Configuration (externalize word lists), Architecture (dynamic word discovery)

---

### PS-4: Humor Subjectivity Has No Ground Truth

**What happens:** Unlike math (where 2+2=4 is verifiable), humor is subjective.
"humor_rating" in the current system is either self-assessed by the model or
manually labeled. Neither provides a reliable, scalable reward signal.

**Warning signs:**
- Model self-rates all its outputs as 8/10 or higher
- Manual labeling is slow, inconsistent between labelers, and not scalable
- Training on subjective humor ratings leads to model learning labeler bias,
  not universal humor

**Prevention:**
- Separate verifiable aspects (phonetic similarity, format compliance) from
  subjective aspects (humor quality)
- Use phonetic correctness as the primary RLVR reward (it's verifiable)
- Use humor quality as a secondary signal via DPO (preference pairs) or
  LLM-as-judge, not as RLVR reward
- For RLVR, define humor-adjacent verifiable checks:
  - Does the parody create a valid English phrase? (verifiable via dictionary lookup)
  - Does the parody change the meaning from the original? (verifiable via embedding similarity)
  - Does the parody use a word from an unexpected category? (verifiable via word list lookup)
- Collect multiple human ratings per example and use agreement as quality signal

**Phase:** Architecture (reward decomposition), Data Generation (labeling strategy)

---

### PS-5: Duplicate Phonetic Logic Across Files

**What happens:** `word_phone.py` and `parody_suggestions.py` both implement phonetic
analysis with overlapping but different logic. Bug fixes or improvements must be
applied in both places, and inconsistencies between them cause confusing behavior.

**Warning signs:**
- Same word pair gets different similarity scores from different tools
- Bug fixed in one file but not the other
- Vowel groupings, stress handling, or rhyme detection differ subtly

**Prevention:**
- Extract shared phonetic logic into a single `phonetics.py` module
- Both tools should import from the shared module
- Write unit tests for the shared module with known word pairs
- Document the scoring algorithm in one place

**Phase:** Architecture (code deduplication)

**Existing codebase issue:** Specific inconsistencies found:
- `word_phone.py._calculate_similarity()`: rhyme=60%, length=30%, stress=10%
- `parody_suggestions.py._calculate_similarity()`: rhyme=40%, primary_vowel=15%,
  near_rhyme=15%, special_pattern=15%, length=5%, stress=5%, front_consonant=5%
- `word_phone.py._get_last_syllable()`: uses vowel group membership check
- `parody_suggestions.py._get_last_syllable()`: uses AEIOU character check
  with primary stress preference
- Both have `VOWEL_REF` string with same groups but different parsing approaches

---

### PS-6: Style Guide / Known Parodies Leak into Training

**What happens:** The generation prompt includes known funny parodies as examples.
If the model simply copies or closely paraphrases these examples, the training data
contains the "answer key" rather than genuine creative output.

**Warning signs:**
- Generated parodies match known100.csv entries exactly
- Model outputs the example parodies instead of creating new ones
- High overlap between training data and the few-shot examples

**Prevention:**
- Exclude the target title's known parody from the examples shown to the model
- Use different known parodies for training generation vs. evaluation
- Track and filter out generated parodies that match known examples
- Rotate which examples are shown to encourage diversity
- Include a decontamination step that removes near-duplicates of known parodies

**Phase:** Data Generation (decontamination), Architecture (prompt construction)

---

## Prevention Summary

Quick reference: pitfall, prevention strategy, and which implementation phase
should address it.

| ID | Pitfall | Prevention | Phase |
|----|---------|-----------|-------|
| CP-1 | Reward hacking via phonetic gaming | Composite rewards, diversity penalties | Architecture |
| CP-2 | Diversity collapse in training data | Forward-KL divergence, curriculum learning | Training |
| CP-3 | smolagents code parsing failures | Output sanitization, stream=False, max_steps | Architecture |
| CP-4 | Verifier false positives/negatives | Calibrate against known-good, noise correction | Data Generation |
| CP-5 | RLVR trains speed not capability | Difficulty-filtered data, SFT first for new skills | Data Generation |
| RL-1 | Training data contamination | Clean eval sets, synthetic titles, tool usage tracking | Evaluation |
| RL-2 | Reward granularity mismatch | Continuous scores, multi-component rewards | Architecture |
| RL-3 | GRPO dataset format mismatch | Standardize on TRL format from the start | Architecture |
| RL-4 | Entropy collapse during training | Monitor entropy, curriculum, robust baselines | Training |
| SA-1 | Infinite agent loops | max_steps limit, consecutive error detection | Architecture |
| SA-2 | Tool import authorization errors | Correct authorized_imports list, pre-load tools | Architecture |
| SA-3 | smolagents version instability | Pin version, abstraction layer, integration tests | Architecture |
| SA-4 | Output capture fragility | Flexible extraction, official telemetry, fallbacks | Architecture |
| SA-5 | Hub tool loading network dependency | Vendor tools locally, health checks | Architecture |
| LB-1 | Response format inconsistency | Model adapter interface, multi-provider testing | Architecture |
| LB-2 | Token limit mismatches | Dynamic prompt truncation, token budget awareness | Architecture |
| LB-3 | Rate limiting varies by provider | Token-aware limiting, backoff, caching | Architecture |
| LB-4 | Error handling returns invalid input | Raise exceptions, retry in wrapper | Architecture |
| DP-1 | Missing config_name in HF YAML | Use push_to_hub(), validate YAML | Dataset Pipeline |
| DP-2 | JSONL vs Parquet decision | JSONL for dev, Parquet for prod | Dataset Pipeline |
| DP-3 | Schema drift | Single schema definition, versioning, migration | Architecture |
| DP-4 | CSV output fragility | JSONL primary format, CSV for viewing only | Architecture |
| DP-5 | No deduplication | Unique IDs, dedup before training, idempotency | Data Generation |
| PS-1 | CMU dictionary OOV words | g2p fallback, expand custom_phones | Architecture |
| PS-2 | Scoring mismatches human perception | Calibrate against judgments, holistic scoring | Architecture |
| PS-3 | Static funny words list | Externalize config, dynamic discovery mode | Configuration |
| PS-4 | Humor subjectivity | Separate verifiable from subjective, use DPO for humor | Architecture |
| PS-5 | Duplicate phonetic logic | Shared module, unit tests | Architecture |
| PS-6 | Known parodies leak into training | Decontamination, rotating examples | Data Generation |

---

## Key Architecture Decisions to Make Early

Based on this research, the following decisions should be resolved in Phase 1
(Architecture) to prevent the most pitfalls:

1. **Reward function design:** Continuous multi-component scores, not binary thresholds
2. **Model adapter interface:** Common interface for all LLM providers
3. **Dataset schema:** TRL-compatible format from the start, defined as dataclass
4. **Phonetic library:** Single shared module with one algorithm
5. **Configuration system:** External YAML/JSON for word lists, examples, model config
6. **Error handling:** Exceptions in model wrapper, graceful agent fallback
7. **smolagents pinning:** Pin exact version, wrap all interactions

---

## Sources

### RLVR and Reward Hacking
- [RLVR with Noisy Rewards under Imperfect Verifiers](https://arxiv.org/abs/2510.00915)
- [From Verifiable Dot to Reward Chain (RLVRR)](https://arxiv.org/abs/2601.18533)
- [Reward Hacking Mitigation using Verifiable Composite Rewards](https://arxiv.org/html/2509.15557v1)
- [RLVR Makes Models Faster Not Smarter](https://www.promptfoo.dev/blog/rlvr-explained/)
- [RLVR Overview - Emergent Mind](https://www.emergentmind.com/topics/reinforcement-learning-with-verified-rewards-rlvr)
- [Label Studio - RLVR Guide](https://labelstud.io/blog/reinforcement-learning-from-verifiable-rewards/)

### Diversity Collapse
- [The Choice of Divergence - DPH-RL](https://arxiv.org/abs/2509.07430)
- [Verbalized Sampling for Mode Collapse](https://arxiv.org/html/2510.01171v3)
- [Diversity Collapse in RL](https://www.emergentmind.com/topics/diversity-collapse-in-rl)

### smolagents Issues
- [Partial stop sequences cause SyntaxError (Issue #1851)](https://github.com/huggingface/smolagents/issues/1851)
- [Streaming breaks CodeAgent (Issue #1872)](https://github.com/huggingface/smolagents/issues/1872)
- [Code parsing errors with Ollama (Issue #1251)](https://github.com/huggingface/smolagents/issues/1251)
- [Repeated code snippet invalid errors (Issue #201)](https://github.com/huggingface/smolagents/issues/201)
- [Cerebras models broken after update (Issue #655)](https://github.com/huggingface/smolagents/issues/655)
- [Multi-agent InterpreterError (Issue #640)](https://github.com/huggingface/smolagents/issues/640)
- [Building Good Agents - smolagents docs](https://huggingface.co/docs/smolagents/tutorials/building_good_agents)
- [smolagents Models Reference](https://huggingface.co/docs/smolagents/en/reference/models)

### GRPO and Training Format
- [GRPO Trainer - TRL docs](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [TRL Dataset Formats](https://github.com/huggingface/trl/blob/main/docs/source/dataset_formats.md)
- [Post Training Qwen3 with GRPO](https://pyimagesearch.com/2025/09/08/post-training-qwen3-for-math-reasoning-using-grpo/)
- [GRPO++ Tricks](https://cameronrwolfe.substack.com/p/grpo-tricks)
- [Unsloth RL Guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide)
- [Implementing GRPO in TRL - HF LLM Course](https://huggingface.co/learn/llm-course/en/chapter12/4)

### HuggingFace Hub and Datasets
- [HF Dataset Upload Guide](https://huggingface.co/docs/hub/datasets-upload-guide-llm)
- [convert_to_parquet multi-config bug (Issue #7067)](https://github.com/huggingface/datasets/issues/7067)

### Phonetic Similarity
- [CMU Pronouncing Dictionary](http://www.speech.cs.cmu.edu/cgi-bin/cmudict)
- [Phonetic Similarity Vectors (Allison Parrish)](https://chuck.stanford.edu/chai/data/aparrish/parrish-2017-aaai.pdf)
- [Phonemic Similarity Metrics (Epstein et al.)](https://www.cs.hunter.cuny.edu/~epstein/papers/Phonemic%20Similarity%20Metrics%20to%20Compare%20Pronunciation%20Methods.pdf)

### Rate Limiting and Multi-Provider
- [Rate Limiting in AI Gateway](https://www.truefoundry.com/blog/rate-limiting-in-llm-gateway)
- [LLM Gateway Patterns](https://collabnix.com/llm-gateway-patterns-rate-limiting-and-load-balancing-guide/)
- [Rate Limits for LLM Providers](https://www.requesty.ai/blog/rate-limits-for-llm-providers-openai-anthropic-and-deepseek)
- [Token Limits and Rate Limits in LLM Inference](https://www.typedef.ai/resources/handle-token-limits-rate-limits-large-scale-llm-inference)
