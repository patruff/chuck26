# Phase 6: Inference & Loop Validation - Research

**Researched:** 2026-02-01
**Domain:** vLLM inference serving, OpenAI-compatible API integration, quantized model deployment on RunPod
**Confidence:** HIGH

## Summary

Phase 6 closes the generate-train-serve loop by deploying the DPO fine-tuned model (produced by Phase 5 as `patruff/chuckles-qwen3-32b-dpo` on HuggingFace Hub) via vLLM on RunPod, then validating that the existing `chuckles generate` CLI works against it with zero code changes.

The core technical challenge is serving a 32B-parameter FP16 merged model on affordable GPU hardware. The merged model from Phase 5 is FP16 (~65GB), which does not fit on a single consumer GPU. Three serving strategies exist, listed in order of recommendation: (1) use the official pre-quantized AWQ model from Qwen (`Qwen/Qwen3-32B-AWQ`) as a baseline comparison, then quantize the custom fine-tuned model to AWQ using AutoAWQ or LLM Compressor; (2) use vLLM's built-in `--quantization bitsandbytes` for runtime NF4 quantization of the FP16 model (no pre-quantization step needed, but slower inference); (3) use `--quantization fp8` for FP8 runtime quantization on an A6000 48GB GPU.

The existing `OpenAICompatibleModel` adapter in `model.py` is already a thin wrapper around the `openai.OpenAI` client with configurable `base_url`. It sends `self.model_id` (from `settings.json`'s `model_name`) as the `model` parameter in every chat completion request. vLLM requires the `model` parameter to match either the HuggingFace repo ID it was launched with or the `--served-model-name` override. This means **zero code changes are needed** -- only `settings.json` must be updated with the correct `api_base_url` (vLLM endpoint) and `model_name` (matching vLLM's served model name).

**Primary recommendation:** Deploy on RunPod using BitsAndBytes runtime NF4 quantization for initial testing (simplest path -- no pre-quantization step), then graduate to pre-quantized AWQ for production performance. Use `--served-model-name` on vLLM to decouple the internal model path from the client-facing name.

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| vllm | >= 0.15.0 | OpenAI-compatible inference server | De facto standard for LLM serving. PagedAttention, continuous batching, native Qwen3 support. Supports AWQ/GPTQ/BitsAndBytes/FP8 quantization. |
| openai (Python) | existing | Client library (already in project) | The `OpenAICompatibleModel` adapter already uses this. Zero changes needed. |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| autoawq | latest | Pre-quantize FP16 model to AWQ 4-bit | When creating a pre-quantized AWQ checkpoint for fastest inference. Officially deprecated but still functional and produces better results than LLM Compressor for Qwen3-32B (per GitHub issue vllm-project/llm-compressor#1600). |
| llm-compressor | latest | Alternative AWQ quantization tool (vLLM-native) | When AutoAWQ is no longer functional. Note: quality issues reported with Qwen3-32B W4A16 quantization (issue #1600). |
| huggingface-hub | existing | Model download from Hub | Already in project. vLLM downloads models automatically from Hub. |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| vLLM | SGLang | Competitive performance but smaller ecosystem. vLLM has wider RunPod template support and community adoption. |
| vLLM | Ollama/llama.cpp | Good for local dev but no continuous batching, limited OpenAI API compatibility. |
| AWQ pre-quantization | BitsAndBytes runtime | BitsAndBytes requires no pre-quantization step (much simpler) but is ~2-4x slower at inference (168 tok/s vs 712 tok/s with Marlin). Best for initial testing. |
| AWQ pre-quantization | FP8 runtime quantization | FP8 requires Hopper/Ada Lovelace GPUs for W8A8 acceleration. Ampere (A6000) supports W8A16 only via Marlin kernels. Good on A6000 48GB but not on RTX 4090 24GB. |

**Installation (on RunPod inference pod):**
```bash
# vLLM is pre-installed on RunPod vLLM worker template
# For manual pod setup:
pip install vllm>=0.15.0

# For AWQ pre-quantization (optional, run on a separate machine or the training pod):
pip install autoawq
```

## Architecture Patterns

### Recommended Deployment Structure

```
training/
├── setup_runpod.sh          # Existing: training environment setup
├── train_dpo.py             # Existing: DPO training
├── merge_and_push.py        # Existing: merge LoRA + push to Hub
├── validate_merge.py        # Existing: adapter vs merged quality check
├── setup_inference.sh        # NEW: inference pod setup + vLLM launch
├── quantize_awq.py          # NEW: AWQ quantization script (optional)
└── validate_inference.py     # NEW: end-to-end inference quality test
```

### Pattern 1: vLLM Serving with BitsAndBytes Runtime Quantization (Simplest)

**What:** Serve the FP16 merged model with on-the-fly NF4 quantization. No pre-quantization step needed.
**When to use:** Initial testing, development iteration, when you want to avoid a separate quantization pipeline.
**VRAM:** ~20-24GB for model + ~4-8GB for KV cache = fits on RTX 4090 (24GB, tight) or A6000 (48GB, comfortable).

```bash
vllm serve patruff/chuckles-qwen3-32b-dpo \
  --quantization bitsandbytes \
  --max-model-len 8192 \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name chuckles-qwen3-32b-dpo \
  --gpu-memory-utilization 0.90 \
  --default-chat-template-kwargs '{"enable_thinking": false}'
```

**Important caveat:** vLLM loads the full FP16 model into CPU RAM first, then quantizes to GPU. Needs ~65GB+ system RAM available during startup. After loading, VRAM usage is ~20-24GB.

**Performance:** ~168 tok/s (slower than AWQ's ~712 tok/s with Marlin kernel).

### Pattern 2: vLLM Serving with Pre-Quantized AWQ Model (Best Performance)

**What:** Pre-quantize the merged FP16 model to AWQ 4-bit, push to Hub, then serve the quantized version.
**When to use:** Production serving, when inference speed matters, when serving many requests.
**VRAM:** ~17-18GB model + ~4-6GB KV cache = fits on RTX 4090 (24GB) or A6000 (48GB).

```bash
# Step 1: Quantize (run once, on training pod or machine with enough RAM)
python quantize_awq.py  # Produces patruff/chuckles-qwen3-32b-dpo-awq on Hub

# Step 2: Serve the pre-quantized model
vllm serve patruff/chuckles-qwen3-32b-dpo-awq \
  --max-model-len 8192 \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name chuckles-qwen3-32b-dpo \
  --gpu-memory-utilization 0.90 \
  --default-chat-template-kwargs '{"enable_thinking": false}'
```

**Performance:** ~579-712 tok/s with Marlin kernel acceleration.

### Pattern 3: RunPod Serverless vLLM Worker (Scale to Zero)

**What:** Deploy on RunPod Serverless using the pre-built vLLM worker template with environment variables.
**When to use:** Intermittent usage, cost optimization (pay per second, scale to zero when idle).

**Environment Variables:**
```
MODEL_NAME=patruff/chuckles-qwen3-32b-dpo-awq
QUANTIZATION=awq
MAX_MODEL_LEN=8192
OPENAI_SERVED_MODEL_NAME_OVERRIDE=chuckles-qwen3-32b-dpo
GPU_MEMORY_UTILIZATION=0.90
CUSTOM_CHAT_TEMPLATE=(set to Qwen3 non-thinking template if needed)
```

**Client settings.json configuration:**
```json
{
  "model_name": "chuckles-qwen3-32b-dpo",
  "api_base_url": "https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1",
  "api_key_env_var": "RUNPOD_API_KEY"
}
```

### Pattern 4: settings.json Configuration for vLLM (Zero Code Changes)

**What:** The existing `chuckles generate` CLI works by only updating settings.json.
**Why it works:** `OpenAICompatibleModel` in `model.py` wraps `openai.OpenAI(api_key=..., base_url=...)` and sends `self.model_id` (from settings.json `model_name`) as the `model` parameter. vLLM's OpenAI-compatible API accepts the same request format.

**Critical requirement:** The `model_name` in settings.json MUST match the vLLM server's served model name.

**For RunPod Pod (direct vLLM serve):**
```json
{
  "model_name": "chuckles-qwen3-32b-dpo",
  "api_base_url": "http://<RUNPOD_POD_IP>:8000/v1",
  "api_key_env_var": "VLLM_API_KEY",
  "funny_words_path": "data/funny_words.json",
  "preferences_path": "data/preferences.json",
  "human_examples_path": "data/human_examples.csv"
}
```

**For RunPod Serverless:**
```json
{
  "model_name": "chuckles-qwen3-32b-dpo",
  "api_base_url": "https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1",
  "api_key_env_var": "RUNPOD_API_KEY",
  "funny_words_path": "data/funny_words.json",
  "preferences_path": "data/preferences.json",
  "human_examples_path": "data/human_examples.csv"
}
```

**Note on API key:** vLLM pods typically don't require an API key, but `OpenAICompatibleModel.__init__` raises `ValueError` if the env var is empty. The simplest workaround is to set the env var to a dummy value (e.g., `export VLLM_API_KEY="not-needed"`) and pass `"api_key_env_var": "VLLM_API_KEY"` in settings.json. For RunPod Serverless, use the actual RunPod API key.

### Anti-Patterns to Avoid

- **Anti-Pattern: Using `--max-model-len=40960` on 48GB GPU.** Default Qwen3 context is 40,960 tokens. vLLM allocates KV cache for the full context length upfront. This will OOM on 48GB with a 32B model. Always set `--max-model-len 8192` (parodies are short -- 8K is more than enough).
- **Anti-Pattern: Using greedy decoding (temperature=0).** Qwen3 performance degrades dramatically with greedy decoding -- produces endless repetitions. Always use Temperature=0.7, TopP=0.8, TopK=20 for non-thinking mode. The existing `create_model()` in `model.py` already sets `temperature=0.7`.
- **Anti-Pattern: Serving the FP16 model without quantization on a single GPU.** The FP16 model is ~65GB. Even an A100 80GB will struggle with KV cache. Always quantize for single-GPU serving.
- **Anti-Pattern: Model name mismatch between settings.json and vLLM.** If vLLM is launched with `vllm serve patruff/chuckles-qwen3-32b-dpo` but settings.json has `"model_name": "chuckles-qwen3-32b-dpo"`, the request will be rejected because vLLM expects `model` to be `patruff/chuckles-qwen3-32b-dpo`. Always use `--served-model-name` to control the expected name.
- **Anti-Pattern: Forgetting `enable_thinking=False` at the vLLM level.** If vLLM serves Qwen3 with thinking mode enabled (the default), the model will produce `<think>...</think>` blocks in responses. The smolagents CodeAgent will fail to parse these correctly. Always disable thinking server-wide with `--default-chat-template-kwargs '{"enable_thinking": false}'`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| OpenAI-compatible inference server | Custom Flask/FastAPI wrapper | vLLM `serve` command | vLLM handles batching, KV cache, quantization, streaming, and OpenAI API compatibility out of the box. |
| Model quantization | Custom quantization scripts | AutoAWQ or LLM Compressor for AWQ; vLLM `--quantization bitsandbytes` for runtime | Quantization requires calibration data, kernel optimization, and careful validation. These tools handle it. |
| API adapter for vLLM | New client code or model wrapper | Existing `OpenAICompatibleModel` + settings.json change | The adapter already works with any OpenAI-compatible endpoint. Zero code changes needed. |
| Inference quality comparison | Manual side-by-side prompting | Structured test script with deterministic prompts | Reproducible comparison requires fixed prompts, consistent parameters, and automated scoring. |
| Chat template handling | Custom Jinja template | vLLM `--default-chat-template-kwargs` flag | vLLM reads the model's built-in chat template from the tokenizer config. Just pass `enable_thinking=false`. |

**Key insight:** The entire inference serving layer requires zero new application code. The work is all operational: launch vLLM with the right flags, update settings.json, and validate the loop works end-to-end.

## Common Pitfalls

### Pitfall 1: Model Name Mismatch Between settings.json and vLLM

**What goes wrong:** The `chuckles generate` CLI sends a `model` parameter in every chat completion request (from `settings.json`'s `model_name`). If this doesn't match what vLLM expects, the request is rejected with a 404 or model-not-found error.

**Why it happens:** By default, vLLM uses the full HuggingFace repo ID (e.g., `patruff/chuckles-qwen3-32b-dpo`) as the model name. But settings.json might have a shorter name. Also, vLLM bug #15845 reports that `--served-model-name` sometimes doesn't take effect in responses.

**How to avoid:** Always launch vLLM with `--served-model-name` matching the `model_name` in settings.json. Test with a simple curl command before running the CLI:
```bash
curl http://localhost:8000/v1/models  # Should list the served model name
```

**Warning signs:** HTTP 404 errors, "Model not found" errors, or the CLI hanging on the first generation.

### Pitfall 2: Qwen3 Thinking Mode Producing <think> Blocks

**What goes wrong:** vLLM serves Qwen3 with thinking mode enabled by default. The model wraps reasoning in `<think>...</think>` tags. The smolagents CodeAgent parser does not expect these tags and fails to extract the parody output.

**Why it happens:** Qwen3 defaults to thinking mode ON. During DPO training (Phase 5), thinking was disabled with `enable_thinking=False`. If the inference server doesn't match, the model produces different output format than what the system expects.

**How to avoid:** Two options:
1. Server-wide: `--default-chat-template-kwargs '{"enable_thinking": false}'`
2. Per-request (from client): pass `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` -- but this requires code changes to the model adapter, violating the zero-code-change requirement.

**Recommendation:** Use option 1 (server-side). This maintains the zero-code-change constraint.

### Pitfall 3: BitsAndBytes Runtime Quantization OOM on Load

**What goes wrong:** vLLM with `--quantization bitsandbytes` loads the full FP16 model into CPU RAM first, then quantizes layer-by-layer to GPU. If the RunPod pod has less than ~65GB system RAM, the load fails with OOM before any inference happens.

**Why it happens:** BitsAndBytes does not support streaming quantization -- it needs the full model in CPU memory during the quantization pass. This is a known limitation.

**How to avoid:** Ensure the RunPod pod has at least 96GB system RAM (most GPU pods have this). Alternatively, use a pre-quantized AWQ model which loads directly in 4-bit format (~17GB).

**Warning signs:** Pod crashes during model loading, Python `MemoryError`, or vLLM logs showing "Killed" during weight loading.

### Pitfall 4: vLLM `--max-model-len` Too Large for Available VRAM

**What goes wrong:** vLLM pre-allocates KV cache for `max_model_len` tokens per sequence. Qwen3-32B's default is 40,960 tokens. With AWQ 4-bit weights (~17GB) + KV cache for 40K tokens (~15-20GB), total VRAM exceeds 24GB (RTX 4090) or even 48GB (A6000).

**Why it happens:** vLLM allocates KV cache eagerly based on max_model_len. Unlike inference frameworks that allocate dynamically, vLLM reserves the maximum upfront for scheduling efficiency.

**How to avoid:** Always set `--max-model-len 8192` for parody generation. Parodies are short text -- 8K tokens is far more than needed and keeps VRAM usage reasonable.

**Warning signs:** vLLM startup error: "Not enough memory to allocate for requested max_model_len", or OOM during the first request.

### Pitfall 5: API Key Requirement in OpenAICompatibleModel

**What goes wrong:** The existing `OpenAICompatibleModel.__init__` raises `ValueError` if the API key environment variable is not set or is empty. vLLM pods typically don't require authentication.

**Why it happens:** The model adapter was built for authenticated APIs (Cerebras, Together.ai). vLLM's OpenAI-compatible API doesn't need a real key, but the adapter checks for it.

**How to avoid:** Set a dummy environment variable: `export VLLM_API_KEY="not-needed"`. Point `api_key_env_var` in settings.json to this variable. vLLM ignores the API key value entirely.

**Warning signs:** `ValueError: Environment variable VLLM_API_KEY is not set` before any inference starts.

### Pitfall 6: Inference Quality Validation Without Objective Metrics

**What goes wrong:** "Quality improvement" is declared based on subjective reading of a few examples. There's no reproducible, quantitative way to compare fine-tuned vs base model outputs.

**Why it happens:** Parody quality is inherently subjective. Without a structured evaluation protocol, the validation step becomes hand-wavy.

**How to avoid:** Use the existing phonetic scoring tools from the project. Run identical prompts through base model and fine-tuned model, score each output with the `word_phonetic_analyzer` tool, and compare average scores. Also run the Google Drive review workflow (from Phase 1) on the fine-tuned model outputs.

**Warning signs:** Validation report says "looks better" without numbers.

## Code Examples

### Example 1: Launching vLLM on RunPod (setup_inference.sh)

```bash
#!/bin/bash
# Source: vLLM docs + Qwen3 deployment guide + RunPod docs
set -e

# Configuration
MODEL_ID="patruff/chuckles-qwen3-32b-dpo"
SERVED_NAME="chuckles-qwen3-32b-dpo"
PORT=8000

echo "Installing vLLM..."
pip install vllm>=0.15.0

echo "Launching vLLM server..."
vllm serve "$MODEL_ID" \
  --quantization bitsandbytes \
  --max-model-len 8192 \
  --host 0.0.0.0 \
  --port "$PORT" \
  --served-model-name "$SERVED_NAME" \
  --gpu-memory-utilization 0.90 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --dtype half
```

### Example 2: AWQ Quantization Script (quantize_awq.py)

```python
# Source: AutoAWQ docs + Qwen AWQ quantization guide
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

MODEL_ID = "patruff/chuckles-qwen3-32b-dpo"
QUANT_OUTPUT = "/workspace/chuckles-qwen3-32b-dpo-awq"
HUB_REPO = "patruff/chuckles-qwen3-32b-dpo-awq"

# Load the FP16 merged model
model = AutoAWQForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# AWQ quantization config
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",
}

# Quantize (uses calibration data internally)
model.quantize(tokenizer, quant_config=quant_config)

# Save locally
model.save_quantized(QUANT_OUTPUT)
tokenizer.save_pretrained(QUANT_OUTPUT)

# Push to Hub
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(folder_path=QUANT_OUTPUT, repo_id=HUB_REPO, repo_type="model")
```

### Example 3: settings.json for vLLM endpoint

```json
{
  "model_name": "chuckles-qwen3-32b-dpo",
  "api_base_url": "http://<RUNPOD_POD_IP>:8000/v1",
  "api_key_env_var": "VLLM_API_KEY",
  "funny_words_path": "data/funny_words.json",
  "preferences_path": "data/preferences.json",
  "human_examples_path": "data/human_examples.csv"
}
```

### Example 4: Testing vLLM Endpoint Before Running CLI

```bash
# Verify model is listed
curl http://localhost:8000/v1/models | python3 -m json.tool

# Test a simple chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chuckles-qwen3-32b-dpo",
    "messages": [
      {"role": "user", "content": "Create a parody of The Shawshank Redemption"}
    ],
    "temperature": 0.7,
    "top_p": 0.8,
    "max_tokens": 256
  }'
```

### Example 5: Inference Validation Script Structure (validate_inference.py)

```python
# Compare fine-tuned model vs base model on identical prompts
# Uses existing phonetic scoring tools for objective comparison

from openai import OpenAI

TEST_PROMPTS = [
    "Create a phonetically-sound parody of: 'The Shawshank Redemption'",
    "Create a phonetically-sound parody of: 'Pulp Fiction'",
    "Create a phonetically-sound parody of: 'The Godfather'",
    "Create a phonetically-sound parody of: 'Jurassic Park'",
    "Create a phonetically-sound parody of: 'Fight Club'",
]

def query_model(client, model_name, prompt):
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        top_p=0.8,
        max_tokens=256,
    )
    return response.choices[0].message.content

# Compare base vs fine-tuned
base_client = OpenAI(base_url="<BASE_MODEL_ENDPOINT>/v1", api_key="dummy")
ft_client = OpenAI(base_url="<FINETUNED_MODEL_ENDPOINT>/v1", api_key="dummy")

for prompt in TEST_PROMPTS:
    base_output = query_model(base_client, "Qwen/Qwen3-32B-AWQ", prompt)
    ft_output = query_model(ft_client, "chuckles-qwen3-32b-dpo", prompt)
    # Score both with phonetic tools and compare
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| AutoAWQ for AWQ quantization | LLM Compressor (vLLM-native) | 2025-2026 | AutoAWQ is deprecated but still produces better Qwen3-32B quantization quality (per issue #1600). Use AutoAWQ for now; switch when LLM Compressor improves. |
| `--enable-reasoning --reasoning-parser deepseek_r1` for Qwen3 | `--reasoning-parser qwen3` (dedicated parser) | vLLM 0.9.0+ | Qwen3 now has its own reasoning parser that correctly handles `enable_thinking=False` per-request. No need to use `deepseek_r1` parser. |
| Qwen3-2504 (original, thinking mode default) | Qwen3-Instruct-2507 (non-thinking only) | July 2025 | 2507 variant never produces `<think>` blocks. If available as fine-tuned base, eliminates thinking mode complexity. Our fine-tuned model uses 2504 base, so we still need `enable_thinking=false`. |
| `FastLanguageModel` (old Unsloth API) | `FastModel` (current Unsloth API) | 2025 | Already handled in Phase 5. Mentioned for context. |

**Deprecated/outdated:**
- **AutoAWQ:** Officially deprecated, but still functional and produces better Qwen3 quantization than the replacement (LLM Compressor). Use with caution.
- **`--enable-reasoning --reasoning-parser deepseek_r1`:** Still works for Qwen3 but the dedicated `qwen3` parser is now available in vLLM 0.9.0+.

## Open Questions

1. **Pre-quantized AWQ vs BitsAndBytes runtime: which to use first?**
   - What we know: BitsAndBytes is simpler (no pre-quantization step) but ~4x slower. AWQ requires a separate quantization step but gives much faster inference.
   - What's unclear: Whether the BitsAndBytes runtime quantization produces equivalent quality to AWQ for Qwen3-32B specifically. Benchmark data is for different models.
   - Recommendation: Start with BitsAndBytes for validation (faster to get running), then create AWQ checkpoint for production use.

2. **LLM Compressor quality for Qwen3-32B AWQ**
   - What we know: GitHub issue #1600 reports "almost corrupted" output from LLM Compressor W4A16 on Qwen3-32B. AutoAWQ still produces good results.
   - What's unclear: Whether this has been fixed in newer LLM Compressor releases. The issue is open.
   - Recommendation: Use AutoAWQ despite deprecation. Monitor LLM Compressor issue #1600 for resolution.

3. **RunPod system RAM for BitsAndBytes loading**
   - What we know: BitsAndBytes loads full FP16 model to CPU first (~65GB). RunPod pods typically have ~100-250GB system RAM.
   - What's unclear: Exact system RAM on RunPod RTX 4090 pods (varies by provider in community cloud).
   - Recommendation: Verify with `free -h` on pod before launching. If insufficient RAM, use AWQ pre-quantized model instead.

4. **Objective quality comparison methodology**
   - What we know: Phonetic scoring tools exist in the project. The parody review workflow from Phase 1 exists.
   - What's unclear: What threshold constitutes "measurable quality improvement" -- how many test prompts, what score difference is meaningful.
   - Recommendation: Use at least 20 test prompts. Run each 3 times (different seeds) for statistical robustness. Compare mean phonetic similarity scores. Any positive delta with p < 0.05 is a pass.

## Serving Option Decision Matrix

| Scenario | Quantization | GPU | vLLM Flags | Pros | Cons |
|----------|-------------|-----|-----------|------|------|
| Quick test | `bitsandbytes` | A6000 48GB | `--quantization bitsandbytes` | No pre-quantization step, simplest setup | Slow inference (~168 tok/s), needs ~65GB CPU RAM for loading |
| Development | `bitsandbytes` | RTX 4090 24GB | `--quantization bitsandbytes --max-model-len 4096` | Cheapest GPU option | Very tight on VRAM, limited context, slow |
| Production (pod) | AWQ pre-quantized | RTX 4090 24GB | `(none -- model is already AWQ)` | Fast inference (~712 tok/s with Marlin), small VRAM footprint | Requires separate AWQ quantization step |
| Production (pod) | AWQ pre-quantized | A6000 48GB | `(none)` | Fast inference, generous VRAM for longer context | More expensive than 4090 |
| Production (serverless) | AWQ pre-quantized | RunPod Serverless 4090 | Set via env vars | Pay per second, scale to zero | Cold start latency (~30-60s model loading) |
| High quality | FP8 runtime | A6000 48GB | `--quantization fp8` | Better quality than 4-bit, no pre-quantization | Only Ampere W8A16 (slower than Marlin AWQ) |

## Integration Points (Verified from Codebase)

### model.py (line 49, 89)
- `self.client = OpenAI(api_key=api_key, base_url=api_base_url)` -- base_url comes from settings.json
- `completion_kwargs["model"] = self.model_id` -- model_name comes from settings.json
- `create_model()` sets `temperature=0.7` -- matches Qwen3 non-thinking recommendation

### config.py (lines 36-42)
- Required settings keys: `model_name`, `api_base_url`, `api_key_env_var`, plus data file paths
- No changes needed to config schema for vLLM support

### cli.py (line 118-125)
- `cmd_generate` loads config, creates model, creates agent, generates -- all using settings.json
- No code changes needed anywhere in the pipeline

### Key constraint: `api_key_env_var` must be non-empty
- `model.py` line 43-47: Raises `ValueError` if the env var is not set
- For vLLM (which doesn't need auth), set a dummy value: `export VLLM_API_KEY="not-needed"`

## Sources

### Primary (HIGH confidence)
- [vLLM OpenAI-Compatible Server docs](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/) -- Serve command, API compatibility
- [vLLM Server Arguments](https://docs.vllm.ai/en/latest/configuration/serve_args/) -- `--served-model-name`, `--quantization`, `--max-model-len`, `--default-chat-template-kwargs`
- [vLLM Quantization docs](https://docs.vllm.ai/en/latest/features/quantization/) -- Supported methods (AWQ, BitsAndBytes, FP8, GPTQ)
- [vLLM BitsAndBytes docs](https://docs.vllm.ai/en/stable/features/quantization/bnb/) -- Runtime NF4 quantization, no calibration needed
- [vLLM Reasoning Outputs docs](https://docs.vllm.ai/en/stable/features/reasoning_outputs/) -- `--enable-reasoning`, `--reasoning-parser`, `--default-chat-template-kwargs`
- [Qwen vLLM Deployment Guide](https://qwen.readthedocs.io/en/latest/deployment/vllm.html) -- Qwen3-specific serve commands, thinking mode, AWQ serving
- [Qwen3 Quickstart](https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html) -- Recommended sampling parameters (T=0.7, TopP=0.8 for non-thinking)
- [RunPod vLLM Worker GitHub](https://github.com/runpod-workers/worker-vllm) -- Environment variables, serverless template
- [RunPod vLLM Environment Variables](https://docs.runpod.io/serverless/workers/vllm/environment-variables) -- MODEL_NAME, QUANTIZATION, MAX_MODEL_LEN, etc.
- [RunPod OpenAI API Compatibility](https://docs.runpod.io/serverless/vllm/openai-compatibility) -- Base URL format, API key usage

### Secondary (MEDIUM confidence)
- [vLLM Quantization Benchmarks (JarvisLabs)](https://docs.jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks) -- BitsAndBytes 168 tok/s, Marlin AWQ 712 tok/s, memory savings
- [Qwen3-32B AWQ on HuggingFace](https://huggingface.co/Qwen/Qwen3-32B-AWQ) -- Official pre-quantized AWQ model
- [vLLM GitHub Issue #15845](https://github.com/vllm-project/vllm/issues/15845) -- `served-model-name` not reflected in response model field
- [vLLM GitHub Issue #13257](https://github.com/vllm-project/vllm/issues/13257) -- Model ID mismatch with OpenAI client
- [QwenLM/Qwen3 Issue #1286](https://github.com/QwenLM/Qwen3/issues/1286) -- Setting enable_thinking=False in vLLM
- [LLM Compressor Issue #1600](https://github.com/vllm-project/llm-compressor/issues/1600) -- Qwen3-32B AWQ quality degradation with llm-compressor vs AutoAWQ
- [RunPod Deploying Qwen3 Guide (Medium)](https://medium.com/@mshojaei77/guide-to-deploying-qwen-3-with-vllm-on-runpod-31b9da6642d0) -- Practical deployment walkthrough

### Tertiary (LOW confidence)
- [RunPod Pricing](https://www.runpod.io/pricing) -- GPU hourly rates (change frequently)
- [Muxup: Vendor-recommended LLM parameters](https://muxup.com/2025q2/recommended-llm-parameter-quick-reference) -- Cross-vendor parameter reference

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- vLLM is the clear choice, verified from official docs and RunPod templates
- Architecture: HIGH -- Integration points verified by reading actual codebase (model.py, config.py, cli.py)
- Pitfalls: HIGH -- Model name mismatch, thinking mode, and VRAM issues verified from multiple GitHub issues
- Quantization options: MEDIUM -- BitsAndBytes runtime quantization verified from docs, but real-world performance on this specific model unverified
- AWQ quality (LLM Compressor vs AutoAWQ): MEDIUM -- Known issue with Qwen3-32B, but may be fixed in newer versions

**Research date:** 2026-02-01
**Valid until:** 2026-03-01 (vLLM releases rapidly -- check for new versions monthly)
