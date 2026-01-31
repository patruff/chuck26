# Pitfalls Research

**Domain:** LLM Fine-tuning (DPO/GRPO) + Inference Serving for Parody Generation
**Researched:** 2026-01-31
**Confidence:** MEDIUM-HIGH (multiple sources verified per pitfall; some RunPod-specific items are LOW confidence from single sources)

---

## Critical Pitfalls

Mistakes that cause wasted GPU hours, broken models, or complete restarts.

### Pitfall 1: Merging QLoRA Adapters Into 4-bit Base Produces Degraded Model

**What goes wrong:**
After training with QLoRA (4-bit base + LoRA adapters), merging the LoRA weights back into the 4-bit quantized base model produces a model with significantly worse quality than the adapter-based version. The merged model may perform close to the untrained base model, effectively losing all fine-tuning progress.

**Why it happens:**
QLoRA trains LoRA adapters against a 4-bit quantized base. Naively merging LoRA deltas into those same 4-bit weights introduces compounding quantization errors. The LoRA weights were learned to compensate for the 4-bit representation, but merging them into quantized weights introduces a double-quantization artifact.

**How to avoid:**
Use Unsloth's `save_pretrained_merged("model", tokenizer, save_method="merged_16bit")`. This method downloads the original FP16 base model weights behind the scenes and merges LoRA adapters into those clean weights. Never use `merged_4bit` unless you have a specific reason and understand the tradeoffs.

For vLLM deployment, the correct pipeline is:
1. Train with QLoRA (4-bit base)
2. Save with `save_method="merged_16bit"` (downloads FP16 base, merges LoRA into it)
3. Upload the 16-bit merged model to HuggingFace Hub
4. Serve via vLLM with appropriate quantization (AWQ, GPTQ, or on-the-fly with `--quantization`)

**Warning signs:**
- Model outputs on vLLM are identical to base model (no fine-tuning effect)
- Perplexity of merged model is higher than expected
- vLLM logs show model loading from a 4-bit checkpoint

**Phase to address:**
Model merging and export phase. Build the merge-and-push script correctly from the start.

**Confidence:** HIGH -- Verified across multiple GitHub issues ([unslothai/unsloth#195](https://github.com/unslothai/unsloth/issues/195), [unslothai/unsloth#2516](https://github.com/unslothai/unsloth/issues/2516), [unslothai/unsloth#1089](https://github.com/unslothai/unsloth/issues/1089)) and official Unsloth documentation.

---

### Pitfall 2: Unsloth/TRL/Transformers Version Mismatch Causes Silent Failures

**What goes wrong:**
Unsloth monkey-patches internal methods in both TRL and Transformers. When versions are mismatched, you get either hard crashes (TypeError on method signatures) or silent behavioral differences (different loss computation, missing gradient accumulation accuracy). A well-documented example: `_get_per_token_logps()` signature changed between TRL versions, and Unsloth's cached compiled version called it with the wrong number of arguments.

**Why it happens:**
Unsloth's optimization strategy involves replacing internal trainer methods with custom Triton kernel implementations. These monkey-patches are tightly coupled to specific TRL and Transformers internal APIs, which change between minor versions. RunPod templates may ship with pre-installed versions that don't match.

**How to avoid:**
Pin exact versions in your RunPod setup script. As of January 2026, the compatible set is:
```bash
pip install unsloth>=2026.1.4 trl>=0.25.0 transformers>=4.57.1
```
Always install Unsloth LAST (it patches the others). Clear any `unsloth_compiled_cache/` directory when upgrading. Test with a single training step before committing to a full run.

**Warning signs:**
- `TypeError` mentioning unexpected keyword arguments or wrong positional argument counts
- Loss curves that look qualitatively different from reference examples
- Warning messages about `num_items_in_batch` not being accepted
- Training succeeding but producing nonsensical outputs

**Phase to address:**
Environment setup phase. Create a tested `requirements.txt` with pinned versions and a validation script that runs one training step.

**Confidence:** HIGH -- Multiple GitHub issues confirm this pattern: [unslothai/unsloth#2916](https://github.com/unslothai/unsloth/issues/2916) (GRPO signature mismatch), [unslothai/unsloth#3750](https://github.com/unslothai/unsloth/issues/3750) (transformers API mismatch), [unslothai/unsloth#3527](https://github.com/unslothai/unsloth/issues/3527) (loss differences).

---

### Pitfall 3: GRPO Reward Functions With Wrong Signature or Return Format

**What goes wrong:**
The GRPO reward function crashes, returns NaN, or silently returns wrong values because the function signature does not match what TRL's GRPOTrainer expects. The most common failure modes: (a) not using `**kwargs`, (b) treating `completions` as plain strings when they are `list[list[dict]]` in conversational format, (c) returning rewards with wrong length, (d) accidentally maximizing when you meant to minimize (sign error).

**Why it happens:**
TRL's GRPOTrainer passes `completions` in different formats depending on dataset format. For conversational datasets, completions are `list[list[dict[str, str]]]` -- each completion is a list containing one message dict with "role" and "content" keys. For standard datasets, they are plain strings. The trainer also passes extra dataset columns as kwargs, which crash the function if `**kwargs` is not in the signature.

The sign error is especially insidious: the TRL docs themselves had a documented example (`return [abs(20 - len(completion)) ...]`) that rewarded moving AWAY from the target because GRPO maximizes reward.

**How to avoid:**
For this project's three reward functions (phonetic_scores, tool_usage, structure_preservation), each must:
1. Accept `completions, **kwargs` at minimum
2. Extract text content correctly: `content = completion[0]["content"]` for conversational format
3. Return `list[float]` with exactly one float per completion
4. Ensure higher return value means BETTER (GRPO maximizes)
5. Use `reward_weights` if combining multiple functions with different scales
6. Test each reward function independently before training

```python
# WRONG: treats completions as strings
def phonetic_reward(completions, **kwargs):
    return [score_phonetics(c) for c in completions]

# RIGHT: extracts content from message dict
def phonetic_reward(completions, **kwargs):
    return [score_phonetics(c[0]["content"]) for c in completions]
```

**Warning signs:**
- `TypeError` or `KeyError` during first training step
- All rewards are the same value (indicates parsing failure returning default)
- `frac_reward_zero_std` metric near 1.0
- Reward metrics moving in unexpected direction

**Phase to address:**
GRPO training script phase. Build and unit-test reward functions before running on GPU.

**Confidence:** HIGH -- Verified from [TRL GRPOTrainer official docs](https://huggingface.co/docs/trl/en/grpo_trainer), [TRL Issue #2644](https://github.com/huggingface/trl/issues/2644), and [TRL Issue #2771](https://github.com/huggingface/trl/issues/2771).

---

### Pitfall 4: RunPod Container Storage Loss on Pod Stop/Reboot

**What goes wrong:**
Checkpoints, trained adapters, and merged models saved to container storage (anything outside `/workspace`) are permanently lost when the pod is stopped, rebooted, or preempted (spot instances). Hours of GPU training are wasted.

**Why it happens:**
RunPod has three storage tiers:
- **Container disk** (ephemeral): Lost on stop/reboot. This is the default filesystem.
- **Volume disk** (pod-persistent): Persists across stop/reboot but lost on pod termination. Mounted at `/workspace`.
- **Network volume** (fully persistent): Survives pod termination, can be moved between pods. Replaces `/workspace` when attached.

Most tutorials save models to default paths like `./output/` which lands on container disk. If the pod is stopped for any reason, everything is gone.

**How to avoid:**
1. Always attach a Network Volume when creating training pods
2. Set ALL output paths to `/workspace/` explicitly:
   ```python
   output_dir="/workspace/checkpoints/dpo-run-1"
   ```
3. Save checkpoints frequently (every 100-200 steps for long runs)
4. After training completes, push to HuggingFace Hub immediately as backup:
   ```python
   model.push_to_hub_merged("username/model-name", tokenizer, save_method="merged_16bit", token="...")
   ```
5. For spot instances: set `save_steps` aggressively (every 50 steps) and use `resume_from_checkpoint=True`

**Warning signs:**
- Pod shows "Stopped" status unexpectedly (spot preemption)
- Files you saved earlier are gone after pod restart
- `ls /workspace/` shows empty directory (no network volume attached)

**Phase to address:**
Environment setup phase. Validate storage configuration BEFORE starting any training.

**Confidence:** HIGH -- Verified from [RunPod official documentation on storage types](https://docs.runpod.io/pods/storage/types) and [RunPod network volumes blog](https://www.runpod.io/blog/network-volumes-on-runpod-secure-cloud).

---

### Pitfall 5: Qwen3 Chat Template / Thinking Mode Mismatch During Fine-tuning

**What goes wrong:**
Fine-tuning Qwen3-32B produces a model that ignores the fine-tuning data and continues to behave like the base model. The model's "style adherence rate" is as low as 1%, meaning it learned almost nothing from the training data. Separately, the model may produce `<think>...</think>` blocks when you do not want them, or fail to produce them when you do.

**Why it happens:**
Qwen3 has a dual-mode architecture with "thinking" and "non-thinking" modes controlled by `enable_thinking` in the chat template. The thinking mode wraps reasoning in `<think>` tags. If your training data format does not match the model's expected chat template, the model effectively ignores it.

A documented case: training on 8,000+ samples in ChatML format with `<think>` blocks yielded only 1% style adherence. Switching the template format to DeepSeek-style dramatically improved results, suggesting the template format used during fine-tuning must align with the model's training template.

Additionally, Qwen3 defaults to thinking mode ON. If your DPO chosen/rejected examples do not include `<think>` blocks but the model generates them during training, there is a mismatch between what the model produces and what the loss function compares against.

**How to avoid:**
1. Decide upfront: thinking mode ON or OFF for your use case. For parody generation, non-thinking mode is likely sufficient (creative task, not math/reasoning).
2. Use `tokenizer.apply_chat_template(messages, enable_thinking=False)` consistently in data preprocessing AND inference.
3. Do NOT include `<think>` blocks in historical turns of multi-turn conversations.
4. For DPO: ensure chosen and rejected responses have the same template format.
5. Consider using the Qwen3-2507 non-thinking variant (Instruct model) which does not generate `<think>` blocks at all.
6. For inference with vLLM, pass `"chat_template_kwargs": {"enable_thinking": false}` in API calls, or use `--enable-reasoning` flag with `--reasoning-parser deepseek_r1` if you DO want thinking.
7. Do NOT use greedy decoding with Qwen3 -- it causes performance degradation and endless repetitions. Use Temperature=0.7, TopP=0.8, TopK=20 for non-thinking mode.

**Warning signs:**
- Model outputs include unexpected `<think>` blocks
- Fine-tuned model behaves identically to base model
- Loss decreases during training but evaluation shows no improvement
- Model enters infinite repetition loops

**Phase to address:**
DPO training script phase AND GRPO training script phase. Get the chat template right in data preprocessing.

**Confidence:** HIGH -- Verified from [Qwen3-32B HuggingFace model card](https://huggingface.co/Qwen/Qwen3-32B), [QwenLM/Qwen3 Issue #1718](https://github.com/QwenLM/Qwen3/issues/1718), and [Unsloth Qwen3 docs](https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune).

---

### Pitfall 6: GRPO Training Stuck With Zero Reward Standard Deviation

**What goes wrong:**
GRPO training appears to run but makes no progress. The `frac_reward_zero_std` metric approaches 1.0, meaning for every prompt, all G completions receive the same reward. Since GRPO computes advantages as `(reward - mean) / std`, zero std means zero advantage, zero gradient signal, and no learning.

**Why it happens:**
For this project's parody domain specifically:
- The phonetic reward function may return 0.0 for all completions if the model has not yet learned to produce parseable parody outputs (all fail parsing, all get 0.0)
- Conversely, if the reward thresholds are too loose, all completions pass and all get 1.0
- With small group sizes (G=4 or G=8), the chance of all-same rewards is higher
- The model may produce very similar completions (low diversity), all scoring identically

**How to avoid:**
1. Design reward functions with continuous (non-binary) outputs. Return floats in [0.0, 1.0] that reflect degree of quality, not pass/fail.
2. Use a larger group size (G=16 or higher) for more reward diversity per prompt.
3. Set `scale_rewards="batch"` in GRPOConfig to use batch-level std instead of group-level, which is more robust to same-reward groups.
4. Monitor `frac_reward_zero_std` -- if it exceeds 0.5, training is not learning effectively.
5. Add a small random noise term to rewards as a last resort to prevent exactly-zero std.
6. Ensure the model CAN produce varied completions by using appropriate temperature (0.6-0.8) during generation.
7. Wait at least 300 steps before concluding training is stuck -- loss=0 is normal in early GRPO training.

**Warning signs:**
- `frac_reward_zero_std` near 1.0
- `grad_norm` near 0.0 (no gradient signal)
- `reward/mean` not changing over hundreds of steps
- All logged sample completions look identical

**Phase to address:**
GRPO training script phase. Design rewards to be continuous and test diversity before full training.

**Confidence:** HIGH -- Verified from [TRL GRPOTrainer docs](https://huggingface.co/docs/trl/en/grpo_trainer), [Unsloth RL Guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide), [huggingface/open-r1 Issue #239](https://github.com/huggingface/open-r1/issues/239), and [unslothai/unsloth Issue #2614](https://github.com/unslothai/unsloth/issues/2614).

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Skip chat template validation | Faster iteration | Model ignores fine-tuning, wasted GPU hours | Never for Qwen3 |
| Save only LoRA adapters (not merged) | Faster saves, smaller files | Cannot serve via vLLM easily; must merge at inference time | Only during iterative development, always merge for deployment |
| Use default RunPod container disk | No setup overhead | All work lost on pod stop/preemption | Never for training runs longer than 30 minutes |
| Hardcode HuggingFace token in scripts | Quick to get running | Token in git history, security risk on shared RunPod | Never -- use environment variables or `huggingface-cli login` |
| Skip reward function unit tests | Start GRPO training faster | Silent reward bugs waste entire training runs | Never for custom reward functions |
| Use TRL default hyperparameters | No tuning needed | Suboptimal convergence, potential instability for 32B models | Only for first sanity-check run |
| Train DPO without SFT first | Skip a training phase | DPO struggles to learn from scratch; needs SFT base | Only if Qwen3-32B already follows the expected format well |

---

## Integration Gotchas

Common mistakes when connecting to external services.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| RunPod Network Volume | Creating pod without attaching volume, then losing all work | Always attach network volume at pod creation time; cannot be added later |
| HuggingFace Hub push | Token expired or not set on RunPod; push fails after hours of training | Run `huggingface-cli login` or set `HF_TOKEN` env var in pod template BEFORE training starts |
| vLLM serving after merge | Serving merged 4-bit model instead of 16-bit merged model | Use `merged_16bit`, then optionally re-quantize for serving (AWQ/GPTQ) or let vLLM handle it |
| Unsloth 16-bit merge download | Merge re-downloads FP16 base model every time if output dir not clean | Either pre-download FP16 to HF cache, or ensure merge output dir is clean before each merge |
| DPO dataset from HuggingFace Hub | Dataset columns do not match TRL DPOTrainer expected format (prompt/chosen/rejected) | Map existing dataset columns to expected names; verify with `dataset[0]` before training |
| GRPO dataset from HuggingFace Hub | Extra metadata columns not passed to reward functions | Ensure dataset columns used by reward functions are named consistently; TRL passes all extra columns as kwargs |
| vLLM + existing chucklesPRIME adapter | `OpenAICompatibleModel` adapter assumes specific response format | Test vLLM endpoint with existing adapter before committing to training; ensure chat completion format matches |

---

## Performance Traps

Patterns that work at small scale but fail at training/serving scale.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Training without gradient checkpointing on 32B model | OOM during forward pass | Always enable `gradient_checkpointing=True` for Qwen3-32B on 48GB | Immediately with batch_size > 1 |
| Long sequence lengths without bounds | OOM mid-training on variable-length data | Set `max_seq_length=2048` explicitly; truncate inputs | When a long training example hits the GPU |
| LoRA on attention layers only | Suboptimal quality; measurably worse than full-layer LoRA | Apply LoRA to ALL linear layers: q, k, v, o, gate, up, down | Noticeable in eval quality |
| Not using paged optimizer | OOM spikes during optimizer step | Use `optim="paged_adamw_8bit"` for 32B models | Random crashes at unpredictable training steps |
| GRPO with small group size (G=4) | High variance, reward_zero_std issues, slow convergence | Use G=8 minimum, G=16 preferred | When reward function is noisy or binary-ish |
| Serving FP16 merged model on 24GB GPU | OOM on inference GPU | Quantize to Q4_K_M or AWQ for 24GB GPUs; FP16 needs 80GB+ | When loading the model |
| DPO without reference model memory budget | OOM because DPO loads TWO models (policy + reference) | Account for 2x model memory; use `ref_model=None` with PEFT (TRL handles it) | When starting DPO training |
| GRPO generation + training on same GPU | OOM or NCCL errors | Use vLLM colocate mode with careful `vllm_gpu_memory_utilization` tuning, or separate GPUs | When generation batch is large |

---

## Security Mistakes

Domain-specific security issues for this workflow.

| Mistake | Risk | Prevention |
|---------|------|------------|
| HuggingFace token committed to training scripts | Token leaked in git, anyone can push to your Hub repos | Use `HF_TOKEN` env var or `huggingface-cli login`; add `*.py` token patterns to `.gitignore` review |
| RunPod API key in scripts | Pod management access leaked | Use RunPod dashboard or env vars; never hardcode |
| Training on RunPod community cloud with sensitive data | Community GPUs may have less isolation | Use Secure Cloud for any proprietary training data; Community Cloud is fine for public datasets |
| Leaving stopped pods with network volumes | Ongoing storage charges ($0.20/GB/month) | Terminate pods when done; download checkpoints and delete volumes |

---

## "Looks Done But Isn't" Checklist

Things that appear complete but are missing critical pieces.

- [ ] **DPO training script:** Often missing chat template application to chosen/rejected -- verify `tokenizer.apply_chat_template` is called consistently with `enable_thinking=False`
- [ ] **GRPO reward functions:** Often missing `**kwargs` in signature -- verify each function accepts and ignores extra kwargs
- [ ] **GRPO reward functions:** Often using wrong sign (minimizing when should maximize) -- verify higher reward = better output
- [ ] **Model merge:** Often saves to container disk instead of network volume -- verify output path starts with `/workspace/`
- [ ] **Model merge:** Often skips pushing to Hub -- verify `push_to_hub_merged` runs after `save_pretrained_merged`
- [ ] **vLLM serving:** Often uses wrong chat template -- verify `--chat-template` matches training template exactly
- [ ] **vLLM serving:** Often missing `--enable-reasoning` or `--reasoning-parser` flags for Qwen3 thinking mode -- verify inference mode matches training mode
- [ ] **RunPod setup:** Often missing network volume attachment -- verify `/workspace/` persists across pod stop/start before starting training
- [ ] **DPO dataset:** Often has chosen/rejected in wrong format for TRL -- verify `dataset["train"][0]` shows expected prompt/chosen/rejected structure
- [ ] **GRPO dataset:** Often missing reward function dependencies (phonetic libraries) on RunPod -- verify reward functions can import all dependencies on the pod
- [ ] **Inference endpoint:** Often not tested with existing `OpenAICompatibleModel` adapter -- verify the existing CLI works with vLLM endpoint before declaring inference "done"
- [ ] **LoRA config:** Often applies LoRA to attention only -- verify `target_modules` includes MLP layers (gate, up, down) for optimal quality

---

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Merged model degraded (4-bit merge) | LOW | Re-merge from saved LoRA adapters using `merged_16bit`; only costs time for the merge operation |
| Version mismatch crashes | LOW | Pin versions, clear cache, reinstall; no data lost if checkpoints are on network volume |
| Reward function bug (GRPO) | MEDIUM | Fix function, resume from last checkpoint; but wasted GPU hours are gone |
| Container storage loss | HIGH | Re-run entire training from scratch; no recovery possible if no checkpoints on network volume |
| Chat template mismatch | MEDIUM-HIGH | Must re-run training with correct template; all prior training is wasted |
| DPO chosen/rejected swapped | HIGH | Must re-run training with corrected dataset mapping; all prior training produces opposite of desired behavior |
| GRPO zero-std stuck training | MEDIUM | Adjust reward functions and/or group size; resume from checkpoint with new config |
| OOM on training GPU | LOW | Reduce batch size, enable gradient checkpointing, or switch to larger GPU; no data lost |
| vLLM serving broken | LOW | Debug chat template, quantization, or model format; model weights are fine |

---

## Pitfall-to-Phase Mapping

How roadmap phases should address these pitfalls.

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| QLoRA merge degradation | Merge & Push script | Test merged model inference before pushing; compare output to adapter-based inference |
| Version mismatch | Environment Setup | Run single training step validation script; check Unsloth/TRL/Transformers version harmony |
| GRPO reward signature | GRPO Training Script | Unit test each reward function offline with mock completions; assert correct output shape and range |
| RunPod storage loss | Environment Setup | Verify network volume mounted: `df -h /workspace/`; test file persistence across pod stop/start |
| Qwen3 chat template | DPO/GRPO Data Preprocessing | Inspect tokenized training samples; verify `<think>` presence matches intent |
| GRPO zero-std stuck | GRPO Training Script | Monitor `frac_reward_zero_std` in first 50 steps; abort and fix if > 0.5 |
| DPO dataset format | DPO Training Script | Print `dataset[0]` and verify columns; check `rewards/accuracies` trends in first 100 steps |
| OOM on A6000 | Environment Setup | Run 5-step warmup with target batch size and seq length; confirm peak VRAM fits |
| vLLM serving mismatch | Inference Serving Script | Compare vLLM output to Unsloth inference output on same input; outputs should be semantically similar |
| Forgotten HuggingFace auth | Environment Setup | Test `huggingface-cli whoami` before starting training |
| Inference chat template | Inference Serving Script | Test with existing `chuckles generate` CLI against vLLM endpoint |

---

## Sources

**Official Documentation (HIGH confidence):**
- [Unsloth Qwen3 Fine-tuning Guide](https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune)
- [TRL GRPOTrainer Documentation](https://huggingface.co/docs/trl/en/grpo_trainer)
- [TRL DPOTrainer Documentation](https://huggingface.co/docs/trl/main/en/dpo_trainer)
- [TRL Reward Functions Documentation](https://huggingface.co/docs/trl/en/rewards)
- [Unsloth RL Guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide)
- [RunPod Storage Types Documentation](https://docs.runpod.io/pods/storage/types)
- [Qwen3-32B HuggingFace Model Card](https://huggingface.co/Qwen/Qwen3-32B)

**GitHub Issues (HIGH confidence -- verified bugs/discussions):**
- [unslothai/unsloth#195](https://github.com/unslothai/unsloth/issues/195) -- Merge to 16-bit dequantization behavior
- [unslothai/unsloth#2516](https://github.com/unslothai/unsloth/issues/2516) -- QLoRA adapter merging urgency
- [unslothai/unsloth#1089](https://github.com/unslothai/unsloth/issues/1089) -- QLoRA merge for vLLM best practice
- [unslothai/unsloth#2916](https://github.com/unslothai/unsloth/issues/2916) -- GRPO _get_per_token_logps signature mismatch
- [unslothai/unsloth#3750](https://github.com/unslothai/unsloth/issues/3750) -- Transformers API mismatch
- [unslothai/unsloth#3527](https://github.com/unslothai/unsloth/issues/3527) -- Loss differences between Unsloth and TRL
- [unslothai/unsloth#2614](https://github.com/unslothai/unsloth/issues/2614) -- GRPO loss is 0
- [unslothai/unsloth#3899](https://github.com/unslothai/unsloth/issues/3899) -- GGUF export garbled output
- [huggingface/trl#2578](https://github.com/huggingface/trl/issues/2578) -- DPO chosen/rejected swap bug (confirmed, fixed)
- [huggingface/trl#2644](https://github.com/huggingface/trl/issues/2644) -- GRPO reward function issues
- [huggingface/trl#2832](https://github.com/huggingface/trl/issues/2832) -- GRPO reward function design tips
- [huggingface/open-r1#239](https://github.com/huggingface/open-r1/issues/239) -- GRPO loss starts at 0
- [vllm-project/vllm#22884](https://github.com/vllm-project/vllm/issues/22884) -- Merged LoRA model loses effect in vLLM
- [QwenLM/Qwen3#1718](https://github.com/QwenLM/Qwen3/issues/1718) -- Fine-tuned Qwen3-32B style adherence failure
- [QwenLM/Qwen3#1286](https://github.com/QwenLM/Qwen3/issues/1286) -- Setting enable_thinking=False in vLLM

**Community Resources (MEDIUM confidence):**
- [Phil Schmid: DPO + Synthetic Data 2025](https://www.philschmid.de/rl-with-llms-in-2025-dpo) -- DPO best practices
- [RunPod Network Volumes Blog](https://www.runpod.io/blog/network-volumes-on-runpod-secure-cloud) -- Storage best practices
- [RunPod Cloud GPU Mistakes Guide](https://www.runpod.io/articles/guides/cloud-gpu-mistakes-to-avoid) -- Common RunPod mistakes
- [Weights & Biases: DPO + QLoRA Report](https://wandb.ai/iasai/QLoRA%20+%20DPO/reports/DPO-QLoRA--Vmlldzo1MTQ2ODMw) -- DPO+QLoRA patterns
- [Stephen Diehl: GRPOTrainer Guide](https://www.stephendiehl.com/posts/grpotrainer/) -- GRPO practical walkthrough
- [Modal: GRPO TRL Tutorial](https://modal.com/docs/examples/grpo_trl) -- GRPO coding example
- [PyImageSearch: Qwen3 GRPO Training](https://pyimagesearch.com/2025/09/08/post-training-qwen3-for-math-reasoning-using-grpo/) -- Qwen3 GRPO walkthrough
- [HuggingFace Cookbook: Advanced GRPO Rewards](https://huggingface.co/learn/cookbook/en/trl_grpo_reasoning_advanced_reward) -- Multi-reward GRPO
- [vLLM Forums: 4-bit LoRA Deploy](https://discuss.vllm.ai/t/support-for-deploying-4-bit-fine-tuned-model-with-lora-on-vllm/1186) -- vLLM + LoRA issues

---
*Pitfalls research for: LLM Fine-tuning & Inference Serving (DPO/GRPO on Qwen3-32B with Unsloth + RunPod + vLLM)*
*Researched: 2026-01-31*
