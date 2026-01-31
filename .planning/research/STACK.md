# Stack Research: Fine-Tuning & Inference Serving

**Domain:** LLM fine-tuning (DPO/GRPO) and inference serving
**Researched:** 2026-01-31
**Confidence:** HIGH (versions verified via PyPI, official docs, and multiple sources)

## Context

chucklesPRIME v1.0 generates phonetically-sound parody title datasets. This stack research covers NEW additions needed for:
1. DPO training on Qwen3-32B with 4-bit QLoRA
2. GRPO training with custom reward functions (phonetic similarity, humor scoring)
3. LoRA adapter merging and Hub push
4. Inference serving via vLLM on RunPod

**Existing stack (DO NOT CHANGE):** smolagents, datasets, pronouncing, openai client, rich, flask, huggingface-hub

---

## Recommended Stack

### Training Core

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| unsloth | >= 2026.1.4 | QLoRA fine-tuning accelerator | 2-2.7x faster training, 70% less VRAM vs standard PEFT. Hand-optimized Triton kernels with 0% accuracy loss. Explicitly supports Qwen3-32B with `FastModel.from_pretrained`. Only viable way to QLoRA-train 32B on a single A6000. |
| trl | >= 0.27.1 | DPOTrainer and GRPOTrainer | Official HuggingFace trainers. DPOTrainer supports preference datasets with chat templates. GRPOTrainer supports custom reward functions, multiple reward functions, CISPO/SAPO/GSPO loss types, and vLLM-accelerated generation. |
| transformers | >= 5.0.0 | Base model loading, tokenizer | Required by TRL 0.27.x. GRPOTrainer agent/tool training requires transformers 5.0.0+. Qwen3 architecture natively supported. |
| peft | >= 0.18.1 | LoRA adapter management, merge_and_unload | Required by Unsloth under the hood. Provides `merge_and_unload()` for merging QLoRA adapters into 16-bit base models for vLLM deployment. |
| bitsandbytes | >= 0.49.1 | 4-bit NF4 quantization | Required for QLoRA training. Supports NF4 and double quantization. SM60+ GPUs (A6000/A100 are SM86/SM80). |
| accelerate | >= 1.12.0 | Training orchestration | Required by TRL for distributed training and DeepSpeed integration. Handles device placement and gradient accumulation. |
| torch | >= 2.4.0 | Deep learning framework | Required by Unsloth. SDPA natively integrated. Use whichever version Unsloth pins -- let `pip install unsloth` resolve this. |

### Inference Serving

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| vllm | >= 0.15.0 | OpenAI-compatible inference server | De facto standard for LLM serving. PagedAttention for efficient KV cache, continuous batching. Native Qwen3 support including thinking mode. AWQ/GPTQ quantization for fitting 32B on consumer GPUs. OpenAI-compatible API means zero changes to existing chucklesPRIME client code. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| huggingface-hub | >= 0.20.0 | Model/dataset push to Hub | Already in project. Used to push merged models and LoRA adapters to Hub. |
| datasets | >= 3.0.0 | Dataset loading for training | Already in project. DPO datasets load directly via `load_dataset()`. |
| wandb | >= 0.19.0 | Training metrics logging | Optional but strongly recommended. TRL integrates natively. Track reward curves, loss, gradient norms. Set `report_to="wandb"` in training config. |
| safetensors | >= 0.5.0 | Safe model serialization | Dependency of transformers. Used for `safe_serialization=True` when saving merged models. |

---

## VRAM Budget Analysis

### Training (Qwen3-32B, 4-bit QLoRA via Unsloth)

| Component | VRAM | Notes |
|-----------|------|-------|
| Model weights (4-bit) | ~20 GB | 32.8B params at 4-bit NF4 |
| LoRA adapter (rank 32) | ~1-2 GB | bf16 adapter weights for q,k,v,o,gate,up,down |
| Optimizer states | ~2-3 GB | AdamW states for LoRA params only |
| Activations/gradients | ~4-8 GB | Depends on batch size and seq length |
| **Total (bs=1, seq=2048)** | **~26-30 GB** | **Fits on A6000 (48GB) with headroom** |
| **Total (bs=2, seq=2048)** | **~32-38 GB** | **Still fits on A6000 (48GB)** |

**Unsloth's official table for 32B QLoRA: 26 GB minimum VRAM.**

**GRPO additional overhead:** GRPO generates G completions per prompt (typically G=8-16). With Unsloth's memory-efficient kernels, GRPO on Qwen3-32B 4-bit fits on A6000 48GB at batch_size=1, seq=2048, G=8. For G=16 or longer sequences, use A100 80GB.

### Inference (vLLM, merged model)

| Precision | Weights VRAM | With KV Cache (8K ctx) | Recommended GPU |
|-----------|-------------|----------------------|-----------------|
| FP16 | ~65 GB | ~75-80 GB | 2x A100-80GB (impractical for this project) |
| AWQ 4-bit | ~20 GB | ~24-28 GB | 1x RTX 4090 (24GB) -- tight but works |
| AWQ 4-bit | ~20 GB | ~24-28 GB | 1x A6000 (48GB) -- comfortable |
| GPTQ 4-bit | ~20 GB | ~24-28 GB | 1x RTX 4090 (24GB) -- tight but works |

**Recommendation:** Serve the merged model as AWQ 4-bit on a single GPU. For RTX 4090 (24GB), limit context to 8K tokens via `--max-model-len 8192`. For A6000 (48GB), context up to 16K+ is feasible.

---

## GPU & RunPod Recommendations

### Training Pod

| Setting | Recommendation | Rationale |
|---------|---------------|-----------|
| **GPU** | NVIDIA A6000 48GB | Cost-effective for Qwen3-32B QLoRA. ~$0.29-0.70/hr on RunPod. 48GB gives headroom for GRPO's multi-completion generation. |
| **Alternative GPU** | NVIDIA A100 80GB | Use if GRPO with G=16+ or seq>4096. ~$1.19-1.64/hr on RunPod. Overkill for basic DPO but needed for aggressive GRPO configs. |
| **Template** | `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu` | Latest RunPod PyTorch template. Compatible with Unsloth's CUDA 12.8 support. |
| **Disk** | 100GB+ volume storage | Need space for: base model download (~65GB fp16, converted to 4-bit on-the-fly), checkpoints, merged 16-bit output (~65GB). |
| **Container Disk** | 50GB | For pip packages, temporary files during merging. |

### Inference Pod (or Serverless)

| Setting | Recommendation | Rationale |
|---------|---------------|-----------|
| **GPU** | RTX 4090 24GB or A6000 48GB | Serve AWQ/GPTQ quantized merged model. 4090 is cheapest option that works. |
| **Template** | RunPod vLLM Worker (from Hub) | Pre-built, cached on RunPod machines for fast cold start. OpenAI-compatible API out of the box. |
| **Environment Variables** | `MODEL_NAME=your-hub-id`, `MAX_MODEL_LEN=8192` | Point at your Hub model. Limit context to fit VRAM. |
| **Alternative** | RunPod GPU Pod with manual vLLM | More control. Run `vllm serve your-model --quantization awq --max-model-len 8192 --enable-reasoning --reasoning-parser deepseek_r1`. |

---

## Installation

### Training Environment (on RunPod Pod)

```bash
# Install Unsloth (resolves compatible torch, transformers, peft, bitsandbytes)
pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo

# Install TRL with vLLM support (for GRPO with vLLM-accelerated generation)
pip install "trl[vllm]>=0.27.1"

# Install training monitoring
pip install wandb>=0.19.0

# Install project dependencies (for reward functions that use phonetic analysis)
pip install pronouncing>=0.2.0

# Verify
python -c "from unsloth import FastModel; print('Unsloth OK')"
python -c "from trl import GRPOTrainer, DPOTrainer; print('TRL OK')"
python -c "import vllm; print(f'vLLM {vllm.__version__} OK')"
```

**CRITICAL:** Let `pip install unsloth` resolve torch/transformers/peft versions. Do NOT pin these independently -- Unsloth has specific version coupling and its installer handles this correctly.

### Inference Environment (on RunPod Pod or Serverless)

```bash
# For manual pod setup
pip install vllm>=0.15.0

# Serve the model
vllm serve your-username/chuckles-qwen3-32b-merged \
  --quantization awq \
  --max-model-len 8192 \
  --host 0.0.0.0 \
  --port 8000 \
  --enable-reasoning \
  --reasoning-parser deepseek_r1
```

For RunPod Serverless, use the pre-built vLLM worker template -- no manual installation needed.

---

## Alternatives Considered

| Category | Recommended | Alternative | Why Not the Alternative |
|----------|-------------|-------------|------------------------|
| Training accelerator | Unsloth | Standard PEFT + bitsandbytes | 2-2.7x slower, 70% more VRAM. Qwen3-32B 4-bit QLoRA would still fit on A6000 but with less headroom for GRPO's multi-completion generation. |
| Training accelerator | Unsloth | Axolotl | More config complexity, YAML-driven. Better for multi-GPU setups (which we don't need for 32B QLoRA). Unsloth is simpler for single-GPU QLoRA. |
| Training accelerator | Unsloth | torchtune (PyTorch official) | Excellent for learning, but less memory-efficient than Unsloth. No Triton kernel optimizations. |
| DPO/GRPO trainer | TRL | Custom implementation | TRL is battle-tested, maintained by HuggingFace, has official Unsloth integration. No reason to reimplement. |
| Inference server | vLLM | TGI (HuggingFace) | vLLM has better throughput via PagedAttention, wider Qwen3 support, native AWQ/GPTQ quantization. RunPod has first-class vLLM templates. |
| Inference server | vLLM | Ollama / llama.cpp | Good for local dev, but no OpenAI-compatible API at scale. No continuous batching for multi-user serving. |
| Inference server | vLLM | SGLang | Competitive performance but smaller ecosystem. vLLM has wider RunPod template support and community adoption. |
| Cloud GPU | RunPod | Lambda Labs / Vast.ai | RunPod has pre-built vLLM serverless workers, volume storage for model persistence, and competitive pricing. Lambda often has availability issues. Vast.ai is cheaper but less reliable. |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `AutoGPTQ` for training | Training-only quantization tool; Unsloth + bitsandbytes NF4 is faster and better integrated | `unsloth` with `load_in_4bit=True` |
| `FastLanguageModel` (old Unsloth API) | Deprecated in favor of `FastModel` for newer models including Qwen3 | `FastModel.from_pretrained()` |
| Manual PEFT LoRA setup | Unsloth's `FastModel.get_peft_model()` handles LoRA config with optimized defaults | Let Unsloth configure LoRA |
| Full fine-tuning | Qwen3-32B full FT needs 80GB+ even with Unsloth. QLoRA at rank 32 on all linear layers achieves near-full-FT quality | QLoRA 4-bit with rank 32 |
| Multi-GPU Unsloth (free tier) | Free Unsloth is single-GPU only. Multi-GPU is a paid feature. Unnecessary for 32B QLoRA. | Single A6000 or A100 |
| Training on the inference GPU | Keep training and inference separate. Training needs large VRAM for optimizer states; inference needs VRAM for KV cache. | Separate RunPod pods |
| Saving merged 4-bit models | Merging QLoRA into a 4-bit model loses precision. Always merge to 16-bit, then re-quantize for serving. | `save_method="merged_16bit"` then AWQ quantize |

---

## Stack Patterns by Variant

**If training DPO only (no GRPO):**
- Simpler setup. DPO needs only preference pairs (chosen/rejected), no multi-completion generation.
- Use `DPOTrainer` from TRL with Unsloth model. No vLLM needed during training.
- VRAM: ~26-30GB on A6000 is very comfortable.

**If training GRPO only (no DPO):**
- GRPO generates G completions per prompt, then scores them with reward functions.
- Use `GRPOTrainer` from TRL. Enable vLLM colocate mode for faster generation: `use_vllm=True, vllm_mode="colocate"`.
- VRAM: ~30-40GB depending on G and seq_length. A6000 works for G=8, seq=2048.
- Consider vLLM server mode on a separate GPU for G=16+ (requires 2 GPUs or paid Unsloth multi-GPU).

**If training DPO then GRPO (recommended pipeline):**
- Phase 1: DPO to align the model with preference data from parody generation traces.
- Phase 2: GRPO to further optimize with phonetic reward functions.
- Each phase is a separate training script. Load the DPO-trained adapter as starting point for GRPO.
- OR: Merge DPO adapter, then start GRPO from the merged model.

**If serving on RTX 4090 (24GB):**
- Must use AWQ or GPTQ 4-bit quantization for the merged model.
- Limit context via `--max-model-len 8192` (or 4096 for more concurrent requests).
- Use Unsloth's `save_pretrained_merged("model", tokenizer, save_method="merged_16bit")` then quantize to AWQ.

**If serving on A6000 (48GB):**
- Can use FP8 quantization for better quality, or AWQ for more headroom.
- Context up to 16K+ tokens is feasible.
- More concurrent requests possible due to larger KV cache budget.

---

## Version Compatibility Matrix

| Package | Min Version | Tested With | Compatibility Notes |
|---------|-------------|-------------|---------------------|
| unsloth | 2026.1.4 | Latest pip | Pins its own torch/transformers/peft versions. Install FIRST. |
| trl | 0.27.1 | transformers 5.0.0 | GRPOTrainer tool support needs transformers >= 5.0.0 |
| transformers | 5.0.0 | peft 0.18.1 | Major version bump from 4.x. Qwen3 natively supported. |
| peft | 0.18.1 | transformers 5.0.0 | `merge_and_unload()` works with QLoRA adapters (must reload base in fp16 first). |
| bitsandbytes | 0.49.1 | CUDA 12.8 | SM60+ required. A6000 (SM86) and A100 (SM80) both supported. |
| accelerate | 1.12.0 | transformers 5.0.0 | Required for `accelerate launch` with TRL trainers. |
| vllm | 0.15.0 | Qwen3, AWQ/GPTQ | >= 0.11.0 for Qwen3-VL, >= 0.9.0 for reasoning parser. 0.15.0 is latest. |
| torch | 2.4.0+ | CUDA 12.8 | Let Unsloth installer resolve. Supports 2.1.0 through 2.9.1. |
| wandb | 0.19.0 | trl 0.27.1 | Optional. Set `report_to="wandb"` in training config. |

**CRITICAL INSTALL ORDER:**
1. Start with RunPod PyTorch template (provides torch + CUDA)
2. `pip install unsloth unsloth_zoo` (resolves transformers, peft, bitsandbytes)
3. `pip install "trl[vllm]>=0.27.1"` (adds TRL + vLLM for GRPO generation)
4. `pip install wandb pronouncing` (monitoring + phonetic reward functions)

---

## Key API Patterns

### Loading Qwen3-32B for Training (Unsloth)

```python
from unsloth import FastModel

model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/Qwen3-32B-unsloth-bnb-4bit",  # Pre-quantized 4-bit
    max_seq_length=2048,
    load_in_4bit=True,
    full_finetuning=False,
)

model = FastModel.get_peft_model(
    model,
    r=32,                    # LoRA rank
    target_modules=[         # All linear layers
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=32,
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",
)
```

### DPO Training

```python
from trl import DPOConfig, DPOTrainer
from datasets import load_dataset

dataset = load_dataset("your-username/chuckles-dpo-pairs", split="train")

training_args = DPOConfig(
    output_dir="./dpo-output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-5,
    bf16=True,
    logging_steps=10,
    report_to="wandb",
)

trainer = DPOTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=dataset,
)
trainer.train()
```

### GRPO Training with Custom Reward

```python
from trl import GRPOConfig, GRPOTrainer

def phonetic_reward(completions, **kwargs):
    """Custom reward: score based on phonetic similarity of generated parody titles."""
    rewards = []
    for completion in completions:
        # Parse the parody title from completion
        # Score using pronouncing library (rhyme/syllable match)
        score = compute_phonetic_score(completion)
        rewards.append(float(score))
    return rewards

training_args = GRPOConfig(
    output_dir="./grpo-output",
    per_device_train_batch_size=1,
    num_generations=8,           # G completions per prompt
    max_completion_length=512,
    learning_rate=1e-6,
    bf16=True,
    report_to="wandb",
    # Enable vLLM for faster generation (colocate mode)
    use_vllm=True,
    vllm_mode="colocate",
    vllm_gpu_memory_utilization=0.4,  # Share GPU with training
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[phonetic_reward, format_reward],
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
```

### Saving & Merging

```python
# Save LoRA adapter (small, ~100MB)
model.save_pretrained_merged("./adapter", tokenizer, save_method="lora")

# Merge into 16-bit for vLLM deployment
model.save_pretrained_merged("./merged-16bit", tokenizer, save_method="merged_16bit")

# Push merged model to Hub
model.push_to_hub_merged(
    "your-username/chuckles-qwen3-32b-merged",
    tokenizer,
    save_method="merged_16bit",
    token="hf_..."
)
```

### Serving with vLLM

```bash
# Serve merged model (quantized at runtime with AWQ)
vllm serve your-username/chuckles-qwen3-32b-merged \
  --quantization awq \
  --max-model-len 8192 \
  --host 0.0.0.0 \
  --port 8000

# Client code (unchanged from existing chucklesPRIME)
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
)
response = client.chat.completions.create(
    model="your-username/chuckles-qwen3-32b-merged",
    messages=[{"role": "user", "content": "Generate a parody of 'The Shawshank Redemption'"}],
)
```

---

## Unsloth vs Standard PEFT: Detailed Comparison

This comparison is critical because Unsloth is the centerpiece of the training stack recommendation.

| Metric | Standard PEFT QLoRA | Unsloth QLoRA |
|--------|---------------------|---------------|
| **Training Speed** | Baseline | 2-2.7x faster |
| **VRAM Usage** | Baseline | 70-74% less |
| **Accuracy vs Full FT** | 80-90% | Claims 0% loss vs baseline LoRA |
| **Dynamic Quantization** | Not available | Down to 1.58-bit with accuracy preservation |
| **HF Ecosystem Compat** | Native | Drop-in compatible (same save/push API) |
| **Multi-GPU** | Yes (via DeepSpeed) | Free tier: single GPU only |
| **GPU Support** | Any CUDA | GTX 1070 through H100 (SM60+) |
| **DPO/GRPO Integration** | Via TRL | Via TRL (official integration documented in TRL docs) |

**Why Unsloth wins for this project:** Single A6000 with 32B model. Unsloth's VRAM savings turn "barely fits" into "comfortable headroom for GRPO." The 2x speed improvement means training runs complete in half the time = half the RunPod cost.

**When to use standard PEFT instead:** Multi-GPU training of 70B+ models, or if Unsloth has a bug with a specific model architecture (check their GitHub issues).

---

## Sources

### PyPI (version verification -- HIGH confidence)

- [TRL PyPI](https://pypi.org/project/trl/) -- v0.27.1, released Jan 24, 2026, Python >= 3.10
- [Unsloth PyPI](https://pypi.org/project/unsloth/) -- v2026.1.4, released Jan 22, 2026, Python 3.9-3.13
- [vLLM PyPI](https://pypi.org/project/vllm/) -- v0.15.0, released Jan 29, 2026, Python 3.10-3.13
- [PEFT PyPI](https://pypi.org/project/peft/) -- v0.18.1, released Jan 9, 2026, Python >= 3.10
- [bitsandbytes PyPI](https://pypi.org/project/bitsandbytes/) -- v0.49.1, released Jan 8, 2026
- [transformers PyPI](https://pypi.org/project/transformers/) -- v5.0.0, released Jan 26, 2026, Python >= 3.10
- [accelerate PyPI](https://pypi.org/project/accelerate/) -- v1.12.0, released Nov 21, 2025

### Official Documentation (HIGH confidence)

- [Unsloth Qwen3 docs](https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune) -- Qwen3-32B loading, fine-tuning steps, model names
- [Unsloth RL Guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide) -- GRPO/GSPO/DR_GRPO setup, reward functions, VRAM requirements
- [Unsloth vLLM deployment](https://unsloth.ai/docs/basics/inference-and-deployment/vllm-guide) -- save_pretrained_merged, merged_16bit, serving workflow
- [Unsloth Requirements](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements) -- VRAM table: 32B QLoRA = 26GB min
- [TRL DPOTrainer docs](https://huggingface.co/docs/trl/en/dpo_trainer) -- v0.27.1 API, dataset format, Unsloth integration section, loss functions
- [TRL GRPOTrainer docs](https://huggingface.co/docs/trl/main/en/grpo_trainer) -- custom reward function API, vLLM colocate/server modes, multi-reward support
- [vLLM Qwen3 recipes](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.html) -- serving config, quantization, thinking mode
- [RunPod vLLM docs](https://docs.runpod.io/serverless/vllm/get-started) -- serverless deployment, vLLM worker templates

### GitHub (HIGH confidence)

- [Unsloth GitHub](https://github.com/unslothai/unsloth) -- model support list, installation, VRAM benchmarks
- [TRL GitHub](https://github.com/huggingface/trl) -- source for DPOTrainer, GRPOTrainer, releases
- [RunPod vLLM worker](https://github.com/runpod-workers/worker-vllm) -- serverless worker source

### Community/Blog (MEDIUM confidence)

- [HuggingFace blog: Unsloth + TRL](https://huggingface.co/blog/unsloth-trl) -- benchmarks showing 1.88x speedup on A100
- [Qwen official Unsloth docs](https://qwen.readthedocs.io/en/latest/training/unsloth.html) -- Qwen team's endorsed Unsloth workflow
- [RunPod PyTorch templates](https://www.runpod.io/articles/guides/pytorch-2-8-cuda-12-8) -- latest template image names
- [Qwen3-32B VRAM analysis](https://apxml.com/models/qwen3-32b) -- inference VRAM breakdowns

---

*Stack research for: chucklesPRIME fine-tuning and inference milestone*
*Researched: 2026-01-31*
