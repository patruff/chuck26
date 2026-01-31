# Features Research

> Research conducted 2026-01-31 for chucklesPRIME RLVR data generation tool.
> Sources cited inline. This document synthesizes findings from RLVR literature,
> dataset pipeline engineering, and phonetic NLP research to inform feature decisions.

---

## Table Stakes Features

These are non-negotiable for any RLVR data generation tool. Without them, the
pipeline produces data that is unreliable, unreproducible, or unusable for training.

### 1. Deterministic Reproducibility

Every generation run must be reproducible given the same inputs. This means:

- **Seed management**: Fix and log random seeds for all LLM calls and any
  stochastic components. Temperature=0 reduces but does not guarantee determinism
  across all stacks. Record the seed in every output record.
  ([Neptune AI: How to Solve Reproducibility in ML](https://neptune.ai/blog/how-to-solve-reproducibility-in-ml))
- **Environment pinning**: Record model name, model version, API endpoint,
  library versions (`smolagents`, `pronouncing`, etc.), and config file checksums
  in every batch run's metadata.
- **Input versioning**: Hash the input CSV and all config files (`funny_words.json`,
  `preferences.json`, `human_examples.csv`) and store hashes alongside outputs.
  If any input changes, the hash changes, signaling the run is not comparable.

### 2. VeRL-Compatible Output Format

The de facto standard for RLVR training data is the **VeRL parquet schema**, which
has been adopted by VeRL, SkyRL, Oumi, and HuggingFace open-instruct. Each record
must contain:

| Column | Type | Description |
|---|---|---|
| `data_source` | `string` | Dataset identifier (e.g., `"chucklesPRIME"`) |
| `prompt` | `list[{role, content}]` | Chat-template formatted prompt |
| `ability` | `string` | Task type (e.g., `"phonetic_parody"`) |
| `reward_model` | `struct{style, ground_truth}` | Reward config + ground truth |
| `extra_info` | `struct{...}` | Metadata (index, phonetic scores, etc.) |

The existing `rlvr_dataset_tools.py` already supports SFT, DPO, RLVR, and tool-use
formats with configurable template tags (default and DeepSeek presets). This is good.
The gap is mapping to VeRL parquet and including reward metadata in `reward_model`.

([VeRL: Prepare Data for Post-Training](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html),
[AllenAI RLVR-IFeval](https://huggingface.co/datasets/allenai/RLVR-IFeval))

### 3. Structured Reasoning Trace Capture

The reasoning trace is the most valuable part of each data point for RLVR training.
The current codebase captures traces via `OutputCapture.callback()` and regex
extraction from `<think>` tags. This needs to be robust:

- **Full multi-step capture**: Every agent step (LLM output, tool calls, tool
  results, action outputs) must be captured in structured form, not just the
  final `<think>` block. The `step_callbacks` mechanism in smolagents is the
  right hook, but the current implementation only saves raw text files.
- **Tool call provenance**: Each tool call must record: tool name, arguments,
  raw result, and the parsed score. This is critical because tool calls ARE
  the verifiable rewards -- they provide the ground truth phonetic scores.
- **OpenTelemetry compatibility**: smolagents supports OpenTelemetry tracing
  natively. Consider instrumenting runs to capture traces in a structured,
  queryable format rather than regex-parsing raw text.

([smolagents OpenTelemetry](https://huggingface.co/docs/smolagents/en/tutorials/inspect_runs),
[GitHub Issue #322: Capturing full CodeAgent thinking](https://github.com/huggingface/smolagents/issues/322))

### 4. Deduplication

Deduplication is essential at multiple levels:

- **Input deduplication**: No duplicate titles in the input CSV. Trivial but
  must be enforced programmatically.
- **Output deduplication**: The same title processed twice may produce the same
  parody. Deduplicate by (input_title, final_parody) pair.
- **Cross-batch deduplication**: When running multiple batches, track all
  previously generated parodies and skip or flag duplicates.
- **Near-duplicate detection**: "The Mattress" and "The Matress" (typo) are
  near-duplicates. Use normalized string comparison.

The DRIVE paper found that deduplication is one of the most impactful data curation
steps, particularly when combined with quality filtering.

([DRIVE: Data Curation Best Practices for RLVR](https://arxiv.org/abs/2511.06307))

### 5. Quality-Gated Auto-Labeling

The existing `auto_label_data_point()` function applies threshold-based filtering
(phonetic score >= 0.6, humor rating >= 6, min tool calls >= 2, reasoning present).
This is a good foundation. Every record must carry:

- A quality label (`good` / `bad` / `marginal`)
- The label source (`auto` / `manual`)
- The specific reason for the label
- All individual scores that contributed to the decision

### 6. Batch Processing with Resume

The current `batch_generate.py` processes titles sequentially with no checkpointing.
If it crashes at title 47 of 100, you lose progress. Required:

- **Checkpoint after each title**: Write results incrementally to JSONL, not
  just at the end.
- **Resume from checkpoint**: On restart, skip already-processed titles.
- **Error isolation**: One failed title must not kill the batch. Log the error
  and continue.

### 7. Logging and Audit Trail

Every run must produce a structured log containing:

- Run ID (UUID), timestamp, git commit hash
- All config file contents or checksums
- Per-title: input, output, all scores, all tool calls, generation time, token count
- Summary statistics: pass rate, average scores, error count

---

## Differentiating Features

These features separate a good RLVR dataset from a mediocre one. They are not
strictly required but dramatically improve training data quality.

### 1. Difficulty-Aware Data Selection

The DRIVE and DEPO papers show that training on examples of **medium difficulty**
(where the model succeeds 30-70% of the time) produces the best RLVR training
signal. Examples that are too easy (always correct) or too hard (always wrong)
contribute zero gradient under GRPO.

For chucklesPRIME, this means:

- **Track per-title difficulty**: Run each title N times (e.g., 3-5) and measure
  success rate. Titles where the model always produces good parodies are "easy"
  and less valuable for training. Titles where it always fails are "too hard."
- **Prioritize medium-difficulty titles**: The sweet spot is titles where the
  model sometimes produces good parodies and sometimes fails.
- **Use difficulty metadata in the dataset**: Include pass@k statistics in
  `extra_info` so downstream training can implement curriculum scheduling.

([DRIVE](https://arxiv.org/abs/2511.06307),
[Online Difficulty Filtering](https://arxiv.org/abs/2504.03380),
[DEPO: High Data Efficiency in RLVR](https://arxiv.org/abs/2509.01321))

### 2. Composite Verifiable Reward Function

Rather than a single binary reward, decompose into multiple verifiable sub-rewards
that are independently checkable. This follows the RLVRR approach of decomposing
rewards into content and style dimensions:

- **Phonetic validity** (binary): Do ALL word replacements score > 0.6?
- **Phonetic quality** (continuous 0-1): Average phonetic similarity score
- **Structural fidelity** (binary): Does the parody have the same word count
  as the original? Same syllable pattern?
- **Tool usage completeness** (binary): Did the agent verify EVERY replacement
  word, or did it skip some?
- **Reasoning quality** (binary): Is a non-trivial reasoning trace present?
- **Format compliance** (binary): Does the output follow the expected structure?

Each sub-reward is stored separately in the dataset so training can weight them.
This also mitigates reward hacking -- a model cannot game one signal without
satisfying the others.

([RLVRR: From Verifiable Dot to Reward Chain](https://arxiv.org/abs/2601.18533),
[Reward Hacking Mitigation using Verifiable Composite Rewards](https://arxiv.org/abs/2509.15557))

### 3. Multi-Generation with Best-of-N Selection

Generate multiple parodies per title (the current prompt asks for 3 attempts)
and select the best. This is analogous to the "rollout budget" in DRIVE:

- Generate N candidates (e.g., 3-5) per title
- Score each candidate on the composite reward
- Select top-2 as the "chosen" outputs
- Optionally retain rejected candidates for DPO pairs

The existing system already asks for multiple attempts in the prompt template.
The improvement is to make this a first-class pipeline feature with structured
selection rather than relying on the model's self-evaluation.

### 4. Provenance-Rich Records

Every output record should carry enough metadata to fully reconstruct how it
was generated:

```json
{
  "provenance": {
    "run_id": "uuid",
    "model": "qwen-3-32b",
    "model_endpoint": "cerebras",
    "temperature": 0.7,
    "input_csv_hash": "sha256:...",
    "config_hashes": {"funny_words": "sha256:...", "preferences": "sha256:..."},
    "generation_timestamp": "2026-01-31T12:00:00Z",
    "generation_duration_ms": 4500,
    "smolagents_version": "1.x.x",
    "tool_versions": {"word_phone": "patruff/word-phone@abc123"}
  }
}
```

### 5. Negative Example Generation

For DPO training, you need paired (chosen, rejected) examples. Currently the
system only generates "good" examples and labels some as "bad" post-hoc. A
deliberate strategy for negative examples:

- **Ablated generation**: Generate parodies WITHOUT phonetic verification tools.
  These will have lower phonetic fidelity and serve as natural negative examples.
- **Perturbation**: Take a good parody and degrade it (swap a well-matched word
  for a poorly-matched one) to create a controlled negative.
- **Low-threshold capture**: Keep parodies that score 0.3-0.5 on phonetics --
  they sound vaguely right but are not good enough.

### 6. Diversity Enforcement

Training data must cover diverse patterns to avoid overfitting:

- **Title length diversity**: Mix 1-word titles ("Jaws"), 2-word ("Kill Bill"),
  3-word ("The Dark Knight"), and longer titles.
- **Genre diversity**: Action, comedy, horror, drama, sci-fi, animation.
- **Transformation pattern diversity**: Single-word swap, double swap, compound
  word modification, homophone swap, prefix/suffix swap.
- **Humor category diversity**: Track which funny_words categories are used
  (bodily_functions, food, animals, etc.) and ensure coverage.
- **Measure diversity**: Use Self-BLEU or n-gram overlap metrics to quantify
  output diversity across the dataset.

---

## Quality Signals for Parody RLVR

These are the specific verifiable rewards that make sense for phonetic parody
generation. They are ordered from most to least objectively verifiable.

### Tier 1: Fully Verifiable (Deterministic, Rule-Based)

These can be computed programmatically with zero ambiguity:

1. **Per-word phonetic similarity score** (continuous 0.0-1.0)
   - Source: `word_phone_tool` comparison between original and replacement word
   - Already implemented in the codebase
   - Threshold: > 0.6 is the current "pass" criterion
   - Enhancement: Use the Smith-Waterman algorithm or Weighted Phoneme
     Substitution Matrix (WPSM) for more nuanced scoring beyond the current
     rhyme/length/stress weighted average
   ([Phonemic Similarity Metrics](https://www.cs.hunter.cuny.edu/~epstein/papers/Phonemic%20Similarity%20Metrics%20to%20Compare%20Pronunciation%20Methods.pdf))

2. **All-words-verified flag** (binary)
   - Did the agent call `word_phone_tool` for EVERY word it changed?
   - If a word was changed without verification, this is a quality failure
   - Detectable by comparing the set of changed words against the set of
     tool-call arguments

3. **Syllable count preservation** (binary)
   - Does the parody have the same number of syllables as the original?
   - Computable directly from CMU dictionary data
   - Important for the parody to "sound right" when spoken aloud

4. **Word count preservation** (binary)
   - Does the parody have the same number of words as the original?
   - Trivial to check

5. **Minimum phonetic threshold** (binary)
   - Are ALL individual word replacement scores >= 0.6?
   - The current `all_phonetic_scores_valid` field

6. **No unchanged output** (binary)
   - Is the parody actually different from the original title?
   - Edge case but must be checked

### Tier 2: Semi-Verifiable (Heuristic, Rule-Based)

These require some heuristic judgment but are still computable:

7. **Funny word usage** (binary/count)
   - Does the parody use at least one word from the `funny_words` list?
   - Not all good parodies use funny words, but usage correlates with humor
   - Track which category the funny word came from

8. **Meaning shift detection** (heuristic)
   - Does the parody create a different semantic meaning from the original?
   - Can be approximated by checking if the replacement word is in a different
     semantic category (e.g., "Matrix" is tech, "Mattress" is furniture)
   - Could use word embeddings to measure semantic distance between original
     and replacement

9. **Title structure preservation** (heuristic)
   - Does the parody maintain articles ("The", "A"), prepositions, and
     conjunctions from the original?
   - "The Matrix" -> "The Mattress" preserves structure
   - "The Matrix" -> "Mattress" loses structure

10. **Reasoning quality score** (heuristic)
    - Does the reasoning trace explain WHY the parody is funny?
    - Does it reference phonetic similarity?
    - Does it compare multiple candidates?
    - Can be scored by checking for presence of key phrases or structured
      sections (Attempt 1, Attempt 2, etc.)

### Tier 3: Soft Verification (Model-Based or Reference-Based)

These cannot be deterministically verified but provide useful signal:

11. **Human example similarity** (reference-based)
    - How similar is the generated parody to the style/quality of human
      examples in `known100.csv`?
    - RLVRR approach: Extract "reward chain" signals from human examples
      (e.g., key transformation patterns, humor categories used)
    - NOT exact matching -- we want new parodies, not copies

12. **Self-rated humor** (model-based)
    - The model's own humor rating (currently requested in prompt as X/10)
    - Known to be unreliable and susceptible to self-serving bias
    - Use only as a weak tiebreaker, never as a primary signal

13. **Transformation pattern classification** (model-based)
    - Categorize the type of transformation used (consonant swap, vowel swap,
      compound word, homophone, prefix change, etc.)
    - Useful for diversity tracking, not quality scoring

### Recommended Composite Reward Formula

For v1, a weighted binary composite:

```
reward = (
    0.40 * all_words_verified +          # Tool discipline
    0.30 * min_phonetic_threshold +       # Phonetic quality
    0.15 * word_count_preserved +         # Structural fidelity
    0.10 * has_reasoning_trace +          # Process quality
    0.05 * uses_funny_word               # Humor signal
)
```

This produces a score in [0.0, 1.0] that is fully deterministic, requires no
LLM judge, and is resistant to reward hacking because gaming one component
(e.g., always using funny words) does not help if phonetic verification fails.

---

## Human Example Integration

The `known100.csv` file contains 100+ human-generated parodies with reasoning.
This is a significant asset. Research suggests several ways to leverage it:

### 1. Few-Shot Prompting (Current Approach -- Keep)

The current system uses `get_example_prompt_text()` to inject 10 examples into
the generation prompt. This is well-supported by research:

- One-Shot RLVR showed that even a single example can dramatically improve
  reasoning model performance (36% to 73.6% on MATH500).
- Few-shot examples establish the "style signature" for the model to follow.
- **Recommendation**: Rotate which examples are shown across runs to increase
  diversity. Currently a fixed first-10 is used.

([One-Shot RLVR](https://arxiv.org/abs/2504.20571))

### 2. Reward Calibration Anchors

Use human examples as calibration points for the reward function:

- **Score all human examples** through the same phonetic verification pipeline.
  This establishes the distribution of scores that "known good" parodies achieve.
- **Set thresholds empirically**: If 90% of human examples score >= 0.55 on
  phonetic similarity, then 0.55 (not 0.6) is the empirically validated threshold.
- **Identify blind spots**: Human examples that score LOW on phonetic metrics but
  are clearly funny (e.g., "Inception" -> "Contraception") reveal where the
  phonetic scoring is insufficient. These cases need special handling or the
  scoring algorithm needs refinement.

### 3. DPO Reference Pairs

Human examples serve as the "chosen" side of DPO pairs:

- Pair each human example (chosen) with a machine-generated inferior version
  (rejected) for the same title.
- This is more reliable than pairing two machine-generated outputs because
  the human examples have genuine humor validation.

### 4. Holdout Evaluation Set

Reserve 20% of human examples as a held-out test set:

- Never show these to the model during generation.
- Use them to evaluate: "Given title X, does the model produce a parody of
  similar quality to the human example?"
- This is the closest thing to a ground truth evaluation for humor.

### 5. Pattern Mining for Reward Design

Analyze the corpus of human examples to extract verifiable patterns:

- What percentage of human parodies change exactly 1 word? 2 words? All words?
- What is the average phonetic similarity in human examples?
- What funny word categories appear most often?
- What transformation types are most common?

These statistics become priors for the reward function. If humans change 1 word
73% of the time, then single-word-change parodies should receive a small bonus
in the reward function.

### 6. Anti-Pattern: Do NOT Use as Exact Match Ground Truth

Human examples should NOT be used as exact-match ground truth (i.e., "the correct
answer for 'The Matrix' is 'The Mattress'"). This would:

- Collapse to supervised learning, eliminating the RL exploration benefit
- Penalize equally-good or better parodies that differ from the human example
- Contradict the creative nature of the task

The RLVRR paper explicitly warns against single-point ground truth for open-ended
generation tasks, proposing reward chains instead.

([RLVRR](https://arxiv.org/abs/2601.18533))

---

## Anti-Features

Things to deliberately NOT build in v1. These are either premature optimizations,
scope creep, or known pitfalls.

### 1. Do NOT Build an LLM-as-Judge Humor Scorer

Using an LLM to rate "how funny" a parody is introduces:

- Non-deterministic rewards (different on each evaluation)
- Self-serving bias (the model rates its own style as funnier)
- Reward hacking vulnerability (the model learns to generate outputs that
  look good to the judge rather than being genuinely funny)
- Computational overhead (loading a reward model during training)

The RLVR literature is clear: **precision over diversity** in rewards. Rule-based
rewards that are precise and consistent outperform model-based rewards that are
noisy and gameable.

([Precision over Diversity](https://arxiv.org/abs/2601.04954),
[Reward Hacking in RLVR](https://www.emergentmind.com/topics/reward-hacking-in-rlvr))

### 2. Do NOT Build Curriculum Scheduling in the Generator

Curriculum scheduling (easy-to-hard ordering) is a training-time concern, not a
data generation concern. The generator should:

- Generate data for ALL titles with full metadata
- Include difficulty signals (pass@k) in the metadata
- Let the training framework (VeRL, TRL, etc.) handle curriculum

Building curriculum logic into the data generator couples generation to training
and makes the data less reusable.

### 3. Do NOT Build Interactive Labeling in v1

The existing `interactive_labeler()` function is nice for exploration but does
not scale. For v1:

- Use auto-labeling exclusively
- Invest time in getting the auto-label thresholds right (calibrated against
  human examples) rather than building interactive UX
- Interactive labeling can be a v2 feature for edge cases

### 4. Do NOT Build Real-Time HuggingFace Hub Pushing

Push to Hub should be a separate, explicit step -- not automatic after each
batch. Reasons:

- Intermediate data may be incomplete or low quality
- Accidental pushes of bad data pollute the dataset
- Push should happen only after quality review and dedup

### 5. Do NOT Build Custom Phonetic Embeddings

The current CMU dictionary + custom pronunciations approach is sufficient.
Building phonetic word embeddings or training custom similarity models is:

- A research project in itself
- Premature optimization when the current approach works
- Better addressed by expanding the custom_phones dictionary for edge cases

### 6. Do NOT Over-Engineer Template Tags

The existing system supports `default` and `deepseek` template presets. Adding
more presets without a concrete training target is premature. When a specific
training framework is chosen, add its template then.

### 7. Do NOT Build Multi-Model Comparison

Running the same title through multiple LLMs and comparing outputs is interesting
but out of scope. v1 should focus on one model (Cerebras/Qwen) and do it well.

---

## Feature Dependencies

This section maps which features depend on which others, to inform implementation
order.

```
Seed Management ──────────────────────────────────────────┐
                                                          v
Input Versioning ─────────────────────────> Provenance-Rich Records
                                                          ^
Environment Pinning ──────────────────────────────────────┘

Structured Trace Capture ─────> Tool Call Provenance ─────> Composite Reward
                                       |                        |
                                       v                        v
                              All-Words-Verified Flag    Quality Auto-Labeling
                                                                |
                                                                v
                                                     VeRL Parquet Output
                                                                |
                                                                v
                                                     Push to Hub (manual)

Human Example Scoring ────────> Threshold Calibration ────> Auto-Label Tuning
        |
        v
Pattern Mining ───────> Diversity Metrics

Batch Processing ─────> Checkpointing ─────> Resume ─────> Cross-Batch Dedup

Multi-Generation ─────> Best-of-N Selection ─────> DPO Pair Construction
        |
        v
Difficulty Tracking ──> pass@k Metadata ──> (Training-side curriculum)
```

### Recommended Implementation Order

**Phase 1: Foundation (Must-have, enables everything else)**
1. Structured trace capture (replace regex with structured step callbacks)
2. Tool call provenance recording
3. Seed management and environment pinning
4. Checkpointed batch processing with resume

**Phase 2: Quality (Makes the data actually useful)**
5. Composite reward function implementation
6. Human example scoring and threshold calibration
7. Quality auto-labeling with calibrated thresholds
8. Input/output deduplication

**Phase 3: Format (Makes the data consumable)**
9. VeRL parquet schema mapping
10. Provenance metadata in every record
11. Push to Hub as explicit CLI step

**Phase 4: Optimization (Makes the data notably better)**
12. Multi-generation with best-of-N selection
13. Difficulty tracking (pass@k per title)
14. Diversity metrics and enforcement
15. Negative example generation for DPO

---

## Sources

- [Label Studio: Reinforcement Learning from Verifiable Rewards](https://labelstud.io/blog/reinforcement-learning-from-verifiable-rewards/)
- [Promptfoo: RLVR Explained](https://www.promptfoo.dev/blog/rlvr-explained/)
- [RLVRR: From Verifiable Dot to Reward Chain (ICLR 2026)](https://arxiv.org/abs/2601.18533)
- [RLVR Implicitly Incentivizes Correct Reasoning](https://arxiv.org/abs/2506.14245)
- [DRIVE: Data Curation Best Practices for RLVR](https://arxiv.org/abs/2511.06307)
- [DEPO: High Data Efficiency in RLVR](https://arxiv.org/abs/2509.01321)
- [Online Difficulty Filtering for Reasoning RL](https://arxiv.org/abs/2504.03380)
- [One-Shot RLVR (NeurIPS 2025)](https://arxiv.org/abs/2504.20571)
- [Precision over Diversity (January 2026)](https://arxiv.org/abs/2601.04954)
- [Reward Hacking Mitigation via Composite Rewards](https://arxiv.org/abs/2509.15557)
- [Reward Hacking in RLVR Systems](https://www.emergentmind.com/topics/reward-hacking-in-reinforcement-learning-with-verifiable-rewards-rlvr)
- [VeRL: Prepare Data for Post-Training](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html)
- [VeRL: Implement Reward Function](https://verl.readthedocs.io/en/latest/preparation/reward_function.html)
- [AllenAI RLVR-IFeval Dataset](https://huggingface.co/datasets/allenai/RLVR-IFeval)
- [smolagents OpenTelemetry Tracing](https://huggingface.co/docs/smolagents/en/tutorials/inspect_runs)
- [smolagents Issue #322: Capturing Full Thinking](https://github.com/huggingface/smolagents/issues/322)
- [Labelbox: How to Create Data for RLVR](https://labelbox.com/blog/how-to-create-data-for-reinforcement-learning-with-verifiable-rewards-rlvr/)
- [Phonemic Similarity Metrics (WPSM)](https://www.cs.hunter.cuny.edu/~epstein/papers/Phonemic%20Similarity%20Metrics%20to%20Compare%20Pronunciation%20Methods.pdf)
- [Phonetic Word Embeddings (LREC 2024)](https://aclanthology.org/2024.lrec-main.1168.pdf)
- [Neptune AI: Reproducibility in ML](https://neptune.ai/blog/how-to-solve-reproducibility-in-ml)
- [Lilian Weng: Reward Hacking in RL](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/)
