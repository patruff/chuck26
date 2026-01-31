# Feature Research: Fine-Tuning & Inference Serving

**Domain:** DPO/GRPO fine-tuning of 32B parameter models and inference serving for a parody generation improvement loop
**Researched:** 2026-01-31
**Confidence:** MEDIUM-HIGH (verified against Unsloth docs, TRL official docs, vLLM docs, RunPod pricing; some version-specific claims are MEDIUM)

---

## Feature Landscape

### Table Stakes (Users Expect These)

Features that must work for training and inference to function at all. Missing any of these means the pipeline does not produce a usable fine-tuned model.

#### DPO Training Features

| Feature | Why Expected | Complexity | Depends On (Existing) | Notes |
|---------|--------------|------------|----------------------|-------|
| 4-bit QLoRA model loading via Unsloth | A 32B model requires ~64GB VRAM at FP16; QLoRA brings this to ~21-24GB, fitting on a single A40 (48GB) or even RTX 4090 (24GB) with small batch sizes | LOW | Existing DPO datasets on Hub | Use `FastLanguageModel.from_pretrained` with `load_in_4bit=True`. Unsloth claims 2x speed, 70% less VRAM vs vanilla PEFT. [Source: Unsloth Qwen3 docs](https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune) |
| LoRA adapter targeting all linear layers | Training only q_proj/v_proj undertargets the model. DPO benefits from updating all linear layers (q, k, v, o, gate, up, down) for style learning | LOW | None | Standard Unsloth pattern: `target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` with rank 16-64. [Source: Unsloth blog + HF TRL blog](https://huggingface.co/blog/unsloth-trl) |
| TRL DPOTrainer integration | DPOTrainer is the standard implementation; Unsloth is fully compatible with TRL's DPOTrainer. Pass `ref_model=None` -- DPOTrainer auto-handles ref by unloading adapter | LOW | Hub-pushed DPO dataset | Use `DPOTrainer` with `DPOConfig`. With Unsloth + QLoRA, pass `ref_model=None` so TRL uses the base model (without adapter) as reference. [Source: TRL DPO docs](https://github.com/huggingface/trl/blob/main/docs/source/dpo_trainer.md) |
| Chat template preservation | The training chat template MUST match the base model's template, and MUST be preserved identically for inference. Mismatch is the #1 cause of "model works in training but garbage in deployment" | LOW | None | Qwen3 uses a specific chat template with `<|im_start|>` / `<|im_end|>` markers. The existing dataset.py already formats prompts as `[{"role": "system", ...}, {"role": "user", ...}]` which is correct. Critical: verify the tokenizer's `chat_template` attribute matches what the dataset uses. [Source: Unsloth vLLM deployment guide](https://docs.unsloth.ai/basics/inference-and-deployment/vllm-guide) |
| Gradient checkpointing | Required to fit training within VRAM budget on 48GB GPUs. Recomputes activations during backward pass instead of storing them | LOW | None | Use `use_gradient_checkpointing="unsloth"` for Unsloth's optimized version that supports very long contexts. [Source: Unsloth docs](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide) |
| HF Hub dataset loading | Training script must load the DPO dataset directly from HuggingFace Hub where the existing pipeline pushes it | LOW | Existing `push_dataset()` in dataset.py | `load_dataset("patruff/chuckles-dpo", split="train")`. The dataset columns must be `prompt`, `chosen`, `rejected` -- already implemented in `build_dpo_dataset()`. |
| Checkpoint saving during training | Training a 32B model takes hours; losing progress to a crash is catastrophic. Save adapter checkpoints every N steps | LOW | None | TRL's `TrainingArguments` supports `save_steps` and `save_total_limit`. Save every 50-100 steps, keep last 3 checkpoints. |

#### GRPO Training Features

| Feature | Why Expected | Complexity | Depends On (Existing) | Notes |
|---------|--------------|------------|----------------------|-------|
| TRL GRPOTrainer with custom reward functions | GRPO requires reward functions, not preference pairs. The existing composite rewards (phonetic quality, structure preservation, tool usage) must be adapted to TRL's reward function interface | MEDIUM | Existing `rewards.py` functions, Hub-pushed GRPO dataset | TRL GRPOTrainer accepts `reward_funcs` as callables with signature `(completions, **kwargs) -> list[float]`. Must adapt existing `compute_phonetic_quality`, `compute_structure_preservation`, `compute_tool_usage_completeness` to this interface. Can pass multiple functions and use `reward_weights`. [Source: TRL GRPO docs](https://huggingface.co/docs/trl/main/grpo_trainer) |
| Multi-generation per prompt | GRPO generates G completions per prompt and computes group-relative advantages. This is core to the algorithm, not optional | LOW (config) | GRPO dataset on Hub | Set `num_generations` in GRPOConfig. Recommended: 8+ generations. This has "virtually no impact on GPU memory" per TRL docs. [Source: TRL GRPO docs](https://huggingface.co/docs/trl/main/grpo_trainer) |
| QLoRA + PEFT integration for GRPO | Same memory constraints as DPO -- must use 4-bit quantization. GRPOTrainer accepts `peft_config` parameter directly | LOW | None | Pass `peft_config=LoraConfig(...)` to GRPOTrainer. When using gradient checkpointing with PEFT, input gradients need to be enabled. QLoRA adapter weights are auto-converted to bf16. [Source: TRL GRPO docs](https://huggingface.co/docs/trl/main/grpo_trainer) |
| Prompt-only dataset format | GRPO uses prompt-only format (no chosen/rejected). The existing `records_to_grpo_dataset()` already produces this with auxiliary columns | LOW | Existing `dataset.py` GRPO format | Dataset needs `prompt` column. Auxiliary columns (`original_title`, `phonetic_scores`, etc.) are passed as kwargs to reward functions. |

#### Model Export Features

| Feature | Why Expected | Complexity | Depends On (Existing) | Notes |
|---------|--------------|------------|----------------------|-------|
| LoRA adapter save to Hub | After training, push the LoRA adapter (~100-300MB) to HuggingFace Hub for versioning and sharing | LOW | HF_TOKEN env var | Use `model.push_to_hub_merged("repo/name", tokenizer, save_method="lora", token="...")`. Small file size makes this fast. [Source: Unsloth docs](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide) |
| Merged 16-bit model save | For vLLM serving, merge LoRA into base model and save as full 16-bit weights. Required for production serving without LoRA overhead | MEDIUM | Trained LoRA adapter | Use `model.save_pretrained_merged("path", tokenizer, save_method="merged_16bit")`. CAUTION: Known bugs with Unsloth merged saves -- always verify output quality after merge. Alternatively, use PEFT's `merge_and_unload()` as fallback. [Source: Unsloth GitHub issues #3146, #611](https://github.com/unslothai/unsloth/issues/3146) |
| Model merge quality validation | After merging LoRA to full model, run inference on a test set to verify the merged model matches adapter-loaded quality | MEDIUM | Merged model, test prompts | This is critical. Multiple Unsloth users report quality degradation after merge. Run 10-20 test prompts through both adapter-loaded and merged models, compare outputs. If quality differs, fall back to serving LoRA adapter on top of base model via vLLM's LoRA serving. |

#### Inference Serving Features

| Feature | Why Expected | Complexity | Depends On (Existing) | Notes |
|---------|--------------|------------|----------------------|-------|
| vLLM OpenAI-compatible server | Serve the fine-tuned model via vLLM with OpenAI-compatible API. This is the standard for production LLM serving | LOW | Merged model or base+LoRA | `vllm serve model_path --dtype auto --max-model-len 4096`. vLLM provides OpenAI-compatible `/v1/chat/completions` endpoint. [Source: vLLM docs](https://docs.vllm.ai/en/latest/) |
| Quantized inference (4-bit/INT8) | Serving Qwen3-32B at FP16 requires ~64GB VRAM. With INT4 quantization, it fits on a single 24GB GPU (~22-24GB). For a 48GB A40 at $0.35/hr, INT8 is fine | LOW | Model on disk/Hub | vLLM supports AWQ, GPTQ, and bitsandbytes quantization. For 48GB GPUs, INT8 gives best quality/memory tradeoff. For 24GB GPUs, use 4-bit. [Source: vLLM docs, RunPod pricing](https://www.runpod.io/pricing) |
| Existing CLI compatibility with vLLM endpoint | The existing `OpenAICompatibleModel` in model.py already supports any OpenAI-compatible endpoint via configurable `base_url`. Point it at vLLM | LOW | Existing `model.py`, `config.py` | Change `settings.json` to point `api_base_url` at the vLLM RunPod endpoint, `api_key_env_var` to a RunPod or custom auth key, and `model_name` to the served model name. Zero code changes needed. |
| Max model length configuration | Control the context window to balance VRAM usage and capability. Limiting to 4K-8K instead of Qwen3's full 40K saves ~40% VRAM | LOW | None | `--max-model-len 4096` or `8192`. For parody generation, prompts are short (<2K tokens typically), so 4096 is more than sufficient. |

---

### Differentiators (Competitive Advantage)

Features that significantly improve the quality of the parody improvement loop but are not strictly required for basic functionality.

| Feature | Value Proposition | Complexity | Depends On | Notes |
|---------|-------------------|------------|------------|-------|
| Composite GRPO reward functions matching existing rewards | Adapt the three existing reward functions (phonetic quality, structure preservation, tool usage completeness) as TRL-compatible GRPO reward functions with configurable weights. This is unique -- no off-the-shelf reward exists for phonetic parody quality | MEDIUM | `rewards.py`, GRPOTrainer | TRL accepts multiple reward funcs as a list with `reward_weights`. Map: `compute_phonetic_quality` -> phonetic reward, `compute_structure_preservation` -> structure reward, `compute_tool_usage_completeness` -> tool usage reward. Each returns `list[float]`. [Source: TRL GRPO custom reward docs](https://huggingface.co/docs/trl/main/grpo_trainer) |
| Full improvement loop automation | Script the complete cycle: generate data -> push to Hub -> train on RunPod -> export model -> serve via vLLM -> generate better data. Manual steps between are friction that kills iteration speed | HIGH | All training + serving features | This is the core value proposition. Each loop iteration should take: ~1hr generate, ~2-4hr train, ~10min export, ~5min deploy. A single script or Makefile orchestrating these steps is the differentiator. |
| vLLM LoRA hot-swapping for A/B testing | Serve multiple LoRA adapters on a single base model, compare fine-tuned vs base model outputs on the same prompts in real-time | MEDIUM | vLLM server, LoRA adapter on Hub | vLLM supports `--enable-lora --max-loras 4 --max-lora-rank 64`. Each request can specify which adapter to use. Enables comparing base model vs DPO-tuned vs GRPO-tuned without restarting server. [Source: Unsloth LoRA hot-swap guide](https://unsloth.ai/docs/basics/inference-and-deployment/vllm-guide/lora-hot-swapping-guide) |
| Training metrics logging and early stopping | Log reward curves, loss, and generation quality during GRPO training. Stop early if reward plateaus to save GPU cost | LOW | GRPOTrainer | TRL logs `reward/mean`, `reward/std`, `completions/mean_length`, `entropy`, `clip_ratio` etc. Integrate with W&B or TensorBoard. Monitor `frac_reward_zero_std` -- if high, all generations are same quality, learning has stalled. [Source: TRL GRPO docs](https://huggingface.co/docs/trl/main/grpo_trainer) |
| Phased training: DPO first, then GRPO | DPO is simpler and more stable; use it first to establish a style baseline from human preferences. Then GRPO with verifiable rewards refines reasoning and phonetic discipline. Research shows DPO is better for "alignment/style" while GRPO excels at "reasoning/structured tasks" | MEDIUM | DPO trained model, GRPO reward funcs | Load DPO-trained LoRA, merge or continue training with GRPO. The DPO phase teaches "human-preferred style"; the GRPO phase teaches "phonetic rigor." This two-stage approach is well-supported by the literature. [Source: DPO vs GRPO comparison research](https://towardsai.net/p/artificial-intelligence/mastering-llm-fine-tuning-grpo-ppo-and-dpo-compared) |
| RunPod serverless deployment | Use RunPod's serverless vLLM worker instead of persistent GPU pod. Pay only for inference time, auto-scales to zero when idle | MEDIUM | Merged model on Hub | RunPod's vLLM worker template accepts model name, auto-deploys. $0.00036/sec for A40 serverless. Ideal for intermittent inference workload (generate parodies for a few hours, then idle). [Source: RunPod vLLM docs](https://docs.runpod.io/serverless/vllm/get-started) |
| Qwen3 thinking mode control | Qwen3 has built-in `/think` and `/no_think` mode switching. For GRPO training, enable thinking to improve reasoning quality. For fast inference, disable thinking for 2-3x speed | LOW | Qwen3 model | Add `/think` or `/no_think` to system prompt. During GRPO training, thinking mode helps the model reason about phonetic choices. During production inference, `/no_think` reduces latency and cost. [Source: Unsloth Qwen3 docs](https://unsloth.ai/blog/qwen3) |

---

### Anti-Features (Commonly Requested, Often Problematic)

Features that seem good but should be deliberately NOT built at this stage.

| Anti-Feature | Why Requested | Why Problematic | Alternative |
|--------------|---------------|-----------------|-------------|
| Full fine-tuning (not LoRA) | "LoRA is a compromise, full FT is better" | A 32B model at full precision requires 672GB+ VRAM for training. Even with DeepSpeed ZeRO-3, needs 4-8x A100 80GB. Cost: $5-11/hr on RunPod vs $0.35/hr for A40 with QLoRA. QLoRA quality for DPO/GRPO is well-established. | Stick with QLoRA. If quality is insufficient after multiple GRPO iterations, consider upgrading to LoRA (no quantization) on 2x A100s before going to full FT. |
| Online GRPO with vLLM colocated generation | "Generate completions during training for true online RL" | Colocating vLLM + training on the same GPU causes memory contention. For 32B models on single-GPU setups, this is impractical. Server mode requires dedicated inference GPU, doubling cost. The training-inference mismatch also introduces gradient bias. | Use offline GRPO: pre-generate completions, compute rewards, then train. Less theoretically pure but more practical for budget-constrained single-GPU training. Or use the simpler DPO approach first which does not require online generation at all. |
| Multi-node distributed training | "Train faster with multiple GPUs" | Adds NCCL configuration complexity, DeepSpeed config tuning, network bandwidth requirements. A 32B QLoRA model fits on a single 48GB GPU. Training time on one A40 is ~2-6 hours for a small dataset (hundreds of examples) which is acceptable for an iteration cycle. | Single GPU A40 ($0.35/hr) or A100 ($1.39/hr). Total training cost per iteration: $1-8. Not worth the engineering overhead of multi-GPU for this scale. |
| Automated hyperparameter search | "Auto-tune learning rate, LoRA rank, batch size" | The search space for RLHF/GRPO is poorly understood. Most successful training runs use well-known defaults (lr=1e-5, rank=16-32, batch=2, grad_accum=4). Hyperparameter search costs 5-10x the training budget. | Use established defaults from Unsloth/TRL documentation. Adjust manually only if training diverges. lr=5e-6 to 2e-5, rank=16-64, batch=2-4 with gradient accumulation to effective batch of 8-16. |
| Custom training framework (not TRL) | "TRL is too heavy, write a custom training loop" | TRL's DPOTrainer and GRPOTrainer handle gradient accumulation, mixed precision, PEFT integration, logging, checkpointing, and hub pushing. Reimplementing this is 1000+ lines of subtle code with known edge cases. | Use TRL. It is the standard, well-tested, and Unsloth is explicitly designed to integrate with it. |
| Production-grade inference autoscaling | "Auto-scale based on request volume" | This is a distributed systems problem, not an ML problem. The project needs inference for batch generation (not user-facing latency), so a single persistent pod or serverless endpoint is sufficient. Building autoscaling infrastructure is pure yak-shaving. | RunPod serverless handles basic scaling. For batch generation, spin up a pod, run the job, terminate. |
| Model evaluation suite (benchmarks, perplexity, etc.) | "Measure model quality with standard benchmarks" | Standard benchmarks (MMLU, HumanEval, etc.) measure general capability. They tell you nothing about parody quality. The parody domain is too niche for standard evals. | Evaluate by running the fine-tuned model through the existing generation pipeline and comparing composite reward scores (phonetic quality, structure preservation) against the base model. 50-100 test titles with before/after comparison is more informative than any benchmark. |
| Wandb/MLflow experiment tracking platform | "Track all experiments centrally" | Adds infrastructure dependency, API key management, and configuration overhead for what is currently a solo-developer project running occasional training jobs. | Use TRL's built-in TensorBoard logging (`--logging_dir`). View with `tensorboard --logdir runs/`. Graduate to W&B only if running more than 5-10 experiments. |

---

## Feature Dependencies

```
[Existing: DPO Dataset on Hub]
    |
    v
[DPO Training Script] -----> [LoRA Adapter Save]
    |                              |
    |                              +-------> [Push Adapter to Hub]
    |                              |
    |                              +-------> [Merge to 16-bit Full Model]
    |                                              |
    |                                              v
    |                                        [Merge Quality Validation]
    |                                              |
    |                                              v
    |                               [vLLM Server Deployment] <--- [Quantized Inference Config]
    |                                              |
    |                                              v
    |                               [Update settings.json base_url] (zero code changes)
    |                                              |
    |                                              v
    |                               [Generate Better Parodies via Existing CLI]
    |                                              |
    |                                              v
    |                               [Convert to New DPO/GRPO Datasets]
    |                                              |
    +----------------------------------------------+  (IMPROVEMENT LOOP CLOSES)

[Existing: GRPO Dataset on Hub]
    |
    v
[Adapt Reward Functions to TRL Interface] -----> [GRPO Training Script]
    |                                                    |
    v                                                    v
[Reward Weights Tuning]                          [LoRA Adapter Save]
                                                        |
                                                   (same export path as DPO)

[DPO Trained Model] ----enhances----> [GRPO Training] (phased training)

[vLLM LoRA Hot-Swap] ----requires----> [vLLM Server + Multiple LoRA Adapters on Hub]
```

### Dependency Notes

- **DPO training requires nothing new in the codebase** -- only a standalone training script and the existing Hub dataset
- **GRPO training requires reward function adaptation** -- the existing `rewards.py` functions must be wrapped in TRL's interface
- **Inference serving requires model export** -- either merged 16-bit or LoRA adapter + base model
- **The improvement loop requires all three** -- training, export, and serving must work before the loop can close
- **Phased training (DPO then GRPO) is optional** -- either method works independently, but combining them is a differentiator
- **vLLM LoRA hot-swap requires only the LoRA adapter path** -- no full merge needed, which sidesteps merge quality bugs

---

## MVP Definition

### Launch With (v1) -- Minimum Viable Training Loop

The minimum to close the improvement loop: generate -> train -> serve -> generate better.

- [ ] **DPO training script for RunPod** -- Standalone Python script using Unsloth + TRL DPOTrainer, loads dataset from Hub, trains QLoRA on Qwen3-32B, saves adapter. Config: 4-bit, rank 32, all linear layers, lr=1e-5, 3 epochs. Why essential: DPO is simpler and more stable than GRPO; validates the entire pipeline first.
- [ ] **LoRA adapter export and Hub push** -- Save trained adapter to Hub. Why essential: Makes the trained model portable and versionable.
- [ ] **Merged 16-bit model export** -- Merge LoRA into base model for vLLM serving. Include a quality validation step (run 20 test prompts, compare to adapter-loaded output). Why essential: vLLM serves merged models fastest; LoRA serving is the fallback.
- [ ] **vLLM inference server setup script** -- Script/instructions to launch vLLM on RunPod serving the merged model with OpenAI-compatible API. Why essential: Closes the loop -- existing CLI can point at vLLM endpoint.
- [ ] **Settings.json template for vLLM endpoint** -- Pre-configured settings.json pointing at a vLLM RunPod endpoint. Why essential: The existing OpenAICompatibleModel in model.py needs zero code changes -- just a config change.
- [ ] **Before/after evaluation script** -- Run 50 test titles through base model and fine-tuned model, compare composite reward scores. Why essential: Proves the training actually improved parody quality.

### Add After Validation (v1.x)

Features to add once the DPO loop is confirmed working.

- [ ] **GRPO training script** -- Adapt reward functions to TRL interface, run GRPO on same dataset. Trigger: DPO loop works but parodies lack phonetic discipline.
- [ ] **Composite reward function wrapper for TRL** -- Package existing rewards.py functions into TRL-compatible format with configurable weights. Trigger: Starting GRPO training.
- [ ] **Phased DPO->GRPO training** -- Load DPO-trained adapter, continue with GRPO. Trigger: Both DPO and GRPO individually show improvement, want combined benefit.
- [ ] **vLLM LoRA hot-swapping** -- Serve multiple adapter versions simultaneously for A/B comparison. Trigger: Multiple trained adapters exist, need to compare.
- [ ] **RunPod serverless deployment** -- Switch from persistent pod to serverless for cost savings during low-usage periods. Trigger: Inference is working but idle GPU costs are noticeable.
- [ ] **Qwen3 thinking mode toggling** -- Enable `/think` during GRPO training, `/no_think` for fast production inference. Trigger: GRPO training is set up and want to optimize.
- [ ] **Training metrics dashboard** -- TensorBoard or W&B integration for monitoring training runs. Trigger: Running 3+ training experiments, need to compare.

### Future Consideration (v2+)

Features to defer until the improvement loop is running and producing measurably better parodies.

- [ ] **Online GRPO with vLLM generation** -- True on-policy training with vLLM generating completions during training. Why defer: Requires dedicated inference GPU, complex memory management, training-inference mismatch handling.
- [ ] **Full model fine-tuning** -- No quantization, train all parameters. Why defer: 10-30x more expensive, requires multi-GPU, unclear if QLoRA quality ceiling has been reached.
- [ ] **Multi-iteration automated pipeline** -- Fully automated loop that runs N iterations of generate->train->deploy without human intervention. Why defer: Need manual validation of quality at each step first.
- [ ] **Model merging (TIES/DARE)** -- Merge multiple LoRA adapters with advanced techniques. Why defer: Need multiple adapters first; simple sequential training may be sufficient.

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| DPO training script (Unsloth + TRL) | HIGH | LOW | P1 |
| LoRA adapter save + Hub push | HIGH | LOW | P1 |
| vLLM server setup | HIGH | LOW | P1 |
| Merged 16-bit export + validation | HIGH | MEDIUM | P1 |
| Settings.json for vLLM endpoint | HIGH | LOW | P1 |
| Before/after evaluation script | HIGH | LOW | P1 |
| GRPO reward function adaptation | HIGH | MEDIUM | P2 |
| GRPO training script | HIGH | MEDIUM | P2 |
| Phased DPO->GRPO training | MEDIUM | MEDIUM | P2 |
| vLLM LoRA hot-swapping | MEDIUM | LOW | P2 |
| RunPod serverless deployment | MEDIUM | MEDIUM | P2 |
| Qwen3 thinking mode control | LOW | LOW | P2 |
| Training metrics logging | LOW | LOW | P3 |
| Full improvement loop automation | MEDIUM | HIGH | P3 |
| Online GRPO with vLLM | LOW | HIGH | P3 |

**Priority key:**
- P1: Must have for first training loop iteration
- P2: Should have, add after DPO loop is validated
- P3: Nice to have, future consideration

---

## Comparable Workflows & Reference Implementations

| Reference | What They Do | Our Approach |
|-----------|-------------|-------------|
| [TRL GRPO cookbook (HuggingFace)](https://huggingface.co/learn/cookbook/en/fine_tuning_llm_grpo_trl) | GRPO training on math reasoning with accuracy reward | Same pattern but with phonetic/structure/tool-usage composite reward |
| [Unsloth DPO Zephyr notebook](https://unsloth.ai/docs/get-started/unsloth-notebooks) | DPO fine-tuning with QLoRA on Zephyr | Adapt for Qwen3-32B with our DPO dataset |
| [Unsloth Qwen3 GRPO notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-GRPO.ipynb) | GRPO with proximity-based reward on Qwen3-4B | Scale to 32B, substitute parody-specific rewards |
| [RunPod vLLM worker](https://github.com/runpod-workers/worker-vllm) | OpenAI-compatible LLM serving on RunPod | Deploy merged model or base+LoRA for inference |
| [TRL GRPO + vLLM online training](https://huggingface.co/learn/cookbook/en/grpo_vllm_online_training) | vLLM-accelerated online GRPO at scale | v2+ consideration; start with offline training |

---

## Hardware & Cost Estimates

Based on RunPod pricing research (verified 2026-01-31):

### Training (DPO or GRPO, QLoRA on Qwen3-32B)

| GPU | VRAM | Hourly Cost | Fits 32B QLoRA? | Est. Training Time | Cost Per Run |
|-----|------|-------------|-----------------|-------------------|-------------|
| A40 | 48GB | $0.35/hr | Yes (comfortable) | 2-4 hrs | $0.70-1.40 |
| RTX A6000 | 48GB | $0.33/hr | Yes (comfortable) | 2-4 hrs | $0.66-1.32 |
| A100 SXM | 80GB | $1.39/hr | Yes (fast, large batch) | 1-2 hrs | $1.39-2.78 |
| RTX 4090 | 24GB | ~$0.44/hr | Tight (batch=1, grad_accum=8) | 3-6 hrs | $1.32-2.64 |

**Recommendation:** A40 at $0.35/hr is the sweet spot -- 48GB is comfortable for 32B QLoRA with batch=2, and it is the cheapest 48GB option.

### Inference (vLLM serving Qwen3-32B)

| GPU | VRAM | Hourly Cost | Quantization | Fits? | Notes |
|-----|------|-------------|-------------|-------|-------|
| A40 | 48GB | $0.35/hr | INT8 | Yes | Good throughput, cheap |
| RTX 4090 | 24GB | ~$0.44/hr | INT4 | Yes | Budget option, lower quality |
| L40S | 48GB | $0.79/hr | INT8 or FP16 | Yes | Faster than A40, 2x cost |

**Recommendation:** A40 at $0.35/hr with INT8 quantization for inference. Total loop cost per iteration (4hr train + 2hr inference): ~$2.10.

---

## Sources

### HIGH Confidence (Official Documentation)
- [TRL GRPOTrainer documentation](https://huggingface.co/docs/trl/main/grpo_trainer) -- Reward function interface, configuration, vLLM integration
- [TRL DPOTrainer documentation](https://github.com/huggingface/trl/blob/main/docs/source/dpo_trainer.md) -- DPO training with PEFT/QLoRA
- [Unsloth Qwen3 fine-tuning guide](https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune) -- Qwen3-specific setup, VRAM, notebooks
- [Unsloth fine-tuning guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide) -- Save methods, merge, Hub push
- [vLLM official documentation](https://docs.vllm.ai/en/latest/) -- Serving, LoRA adapters, quantization
- [RunPod vLLM deployment guide](https://docs.runpod.io/serverless/vllm/get-started) -- Serverless deployment

### MEDIUM Confidence (Verified with Multiple Sources)
- [HuggingFace blog: Unsloth + TRL integration](https://huggingface.co/blog/unsloth-trl) -- Benchmarks, DPO example
- [RunPod GPU pricing](https://www.runpod.io/pricing) -- Hardware costs (verified 2026-01-31)
- [PEFT model merging guide](https://huggingface.co/docs/peft/en/developer_guides/model_merging) -- merge_and_unload(), TIES, DARE
- [DPO vs GRPO comparison](https://towardsai.net/p/artificial-intelligence/mastering-llm-fine-tuning-grpo-ppo-and-dpo-compared) -- When to use each method

### LOW Confidence (Single Source or Training-Data Knowledge)
- QLoRA VRAM estimates for 32B (~21-24GB) -- Based on general scaling rules and community reports, not official Unsloth benchmarks for Qwen3-32B specifically
- Training time estimates (2-4 hrs on A40) -- Extrapolated from smaller model benchmarks; actual time depends heavily on dataset size, sequence length, and batch configuration
- Unsloth merged model quality issues -- Based on GitHub issue reports; may be version-specific and could be resolved in current Unsloth release

---
*Feature research for: DPO/GRPO fine-tuning and inference serving for chucklesPRIME parody improvement loop*
*Researched: 2026-01-31*
