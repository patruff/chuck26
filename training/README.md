# Training Parody Models on RunPod

This guide covers setting up DPO training for parody generation on RunPod with cheap hardware.

## Quick Start

```bash
# On RunPod, after pod starts:
cd /workspace
git clone https://github.com/patruff/chuck26.git
cd chuck26/training

pip install -r requirements.txt

# Train (takes ~1-2 hours on RTX 3090)
python train_dpo.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --dataset patruff/chuckles-dpo \
    --output ./parody-model \
    --epochs 3 \
    --use-4bit

# Evaluate
python evaluate_parodies.py --model ./parody-model
```

---

## Cheap Hardware Recommendations

### Budget Options (< $0.30/hr)

| GPU | VRAM | Cost/hr | Best For | Max Model |
|-----|------|---------|----------|-----------|
| **RTX 3090** | 24GB | ~$0.22 | Best value | Qwen2.5-3B, Llama-3.2-3B |
| RTX 3080 | 10GB | ~$0.15 | Tight budget | Qwen2.5-1.5B only |
| RTX 4080 | 16GB | ~$0.28 | Faster training | Qwen2.5-3B |

### Mid-Range Options ($0.30-0.50/hr)

| GPU | VRAM | Cost/hr | Best For | Max Model |
|-----|------|---------|----------|-----------|
| **RTX 4090** | 24GB | ~$0.44 | Speed + quality | Qwen2.5-7B (4-bit) |
| A40 | 48GB | ~$0.39 | Larger models | Qwen2.5-7B, Llama-8B |
| L40 | 48GB | ~$0.49 | Production | Qwen2.5-14B (4-bit) |

### Recommended Setup: RTX 3090 + Qwen2.5-1.5B

- **Cost**: ~$0.22/hr × 2 hours = **$0.44 total**
- **Model**: Qwen2.5-1.5B-Instruct (smallest, fastest)
- **Training time**: ~1.5-2 hours for 3 epochs
- **Quality**: Good enough for parody generation

---

## Cheap Model Recommendations

| Model | Params | VRAM (4-bit) | VRAM (16-bit) | Quality |
|-------|--------|--------------|---------------|---------|
| **Qwen2.5-1.5B-Instruct** | 1.5B | ~2GB | ~4GB | Good |
| Qwen2.5-3B-Instruct | 3B | ~3GB | ~7GB | Better |
| Llama-3.2-1B-Instruct | 1B | ~1.5GB | ~3GB | Basic |
| Llama-3.2-3B-Instruct | 3B | ~3GB | ~7GB | Good |
| Phi-3-mini-4k-instruct | 3.8B | ~3GB | ~8GB | Good |
| Qwen2.5-7B-Instruct | 7B | ~5GB | ~15GB | Best |

**Recommendation**: Start with **Qwen2.5-1.5B-Instruct** — it's small, fast, and surprisingly good at creative tasks like parodies.

---

## RunPod Setup (Step by Step)

### 1. Create RunPod Account

1. Go to [runpod.io](https://runpod.io)
2. Sign up and add credits ($5-10 is enough for testing)

### 2. Launch a Pod

1. Click "Deploy" → "GPU Pods"
2. Select template: **RunPod Pytorch 2.1** (or similar with CUDA)
3. Select GPU: **RTX 3090** (cheapest with 24GB)
4. Select storage: **20GB** (minimum for model + checkpoints)
5. Click "Deploy"

### 3. Connect and Setup

```bash
# SSH into the pod or use the web terminal
cd /workspace

# Clone repo
git clone https://github.com/patruff/chuck26.git
cd chuck26/training

# Install dependencies
pip install -r requirements.txt

# Login to HuggingFace (for dataset access)
huggingface-cli login
# Paste your HF token when prompted
```

### 4. Start Training

```bash
# Basic training with 4-bit quantization (saves VRAM)
python train_dpo.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --dataset patruff/chuckles-dpo \
    --output ./parody-1.5b-dpo \
    --epochs 3 \
    --use-4bit

# With Weights & Biases logging
export WANDB_API_KEY="your-key"
python train_dpo.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --dataset patruff/chuckles-dpo \
    --output ./parody-1.5b-dpo \
    --epochs 3 \
    --use-4bit \
    --wandb-project parody-training
```

### 5. Evaluate the Model

```bash
# Run automatic evaluation
python evaluate_parodies.py \
    --model ./parody-1.5b-dpo \
    --output eval-report.json

# Compare against baseline
python evaluate_parodies.py \
    --model ./parody-1.5b-dpo \
    --baseline Qwen/Qwen2.5-1.5B-Instruct \
    --output comparison.json
```

### 6. Push to HuggingFace Hub

```bash
# Push model
python train_dpo.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --dataset patruff/chuckles-dpo \
    --output ./parody-1.5b-dpo \
    --push-to-hub \
    --hub-model-id your-username/parody-1.5b-dpo

# Or manually
huggingface-cli upload your-username/parody-1.5b-dpo ./parody-1.5b-dpo
```

---

## Training Parameters

### Recommended Settings by GPU

| GPU | Model | Batch Size | Gradient Accum | Use 4-bit |
|-----|-------|------------|----------------|-----------|
| RTX 3090 | 1.5B | 2 | 4 | Optional |
| RTX 3090 | 3B | 1 | 8 | Yes |
| RTX 4090 | 3B | 2 | 4 | Optional |
| A40 | 7B | 2 | 4 | Optional |

### Full Parameter Reference

```bash
python train_dpo.py --help

# Key parameters:
--model           # Base model (HF ID or local path)
--dataset         # DPO dataset (default: patruff/chuckles-dpo)
--output          # Output directory
--epochs          # Training epochs (default: 3)
--batch-size      # Per-device batch size (default: 2)
--gradient-accumulation  # Gradient accumulation steps (default: 4)
--learning-rate   # Learning rate (default: 5e-5)
--lora-r          # LoRA rank (default: 16)
--lora-alpha      # LoRA alpha (default: 32)
--beta            # DPO beta (default: 0.1, higher = more conservative)
--use-4bit        # Enable 4-bit quantization
--use-8bit        # Enable 8-bit quantization
--push-to-hub     # Push to HF Hub after training
```

---

## Automatic Testing

The evaluation script tests parodies automatically:

### Test Criteria

1. **Phonetic Similarity**: Each word substitution must score ≥ 0.6
2. **Average Score**: Overall average must be ≥ 0.65
3. **Structure**: Word count should match original (±20%)

### Running Tests

```bash
# Basic evaluation (15 default test titles)
python evaluate_parodies.py --model ./parody-model

# Custom test titles
python evaluate_parodies.py \
    --model ./parody-model \
    --titles "The Matrix,Die Hard,Top Gun,Star Wars"

# Set minimum pass rate (default: 70%)
python evaluate_parodies.py \
    --model ./parody-model \
    --min-pass-rate 0.8

# Save detailed report
python evaluate_parodies.py \
    --model ./parody-model \
    --output eval-report.json
```

### CI/CD Integration

```yaml
# Example GitHub Actions step
- name: Evaluate Model
  run: |
    python training/evaluate_parodies.py \
      --model ./parody-model \
      --min-pass-rate 0.7 \
      --output eval-report.json

# Exits with code 1 if pass rate < min-pass-rate
```

### Sample Output

```
Evaluating on 15 titles...
------------------------------------------------------------
✓ PASS  The Matrix                → The Mattress              (avg: 0.82)
✓ PASS  Die Hard                  → Dye Hard                  (avg: 0.95)
✗ FAIL  Fight Club                → Fright Clown              (avg: 0.58)
✓ PASS  Top Gun                   → Top Bun                   (avg: 0.88)
...

============================================================
EVALUATION REPORT: ./parody-model
============================================================
  Total titles:     15
  Passed:           12
  Failed:           3
  Pass rate:        80.0%
  Avg phonetic:     0.756

✓ Evaluation PASSED (pass rate 80.0% >= 70.0%)
```

---

## Complete Training Pipeline

```bash
#!/bin/bash
# Full training + evaluation pipeline

set -e  # Exit on error

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
DATASET="patruff/chuckles-dpo"
OUTPUT="./parody-model"
HUB_ID="your-username/parody-1.5b-dpo"

echo "=== Step 1: Training ==="
python train_dpo.py \
    --model $MODEL \
    --dataset $DATASET \
    --output $OUTPUT \
    --epochs 3 \
    --use-4bit

echo "=== Step 2: Evaluation ==="
python evaluate_parodies.py \
    --model $OUTPUT \
    --baseline $MODEL \
    --output eval-report.json \
    --min-pass-rate 0.7

echo "=== Step 3: Push to Hub ==="
huggingface-cli upload $HUB_ID $OUTPUT

echo "=== Done! ==="
echo "Model: https://huggingface.co/$HUB_ID"
```

---

## Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size
--batch-size 1 --gradient-accumulation 8

# Enable 4-bit quantization
--use-4bit

# Use smaller model
--model Qwen/Qwen2.5-1.5B-Instruct
```

### Slow Training

```bash
# Enable Flash Attention (Ampere+ GPUs)
pip install flash-attn --no-build-isolation

# Use bf16
# (automatically enabled when CUDA is available)
```

### Poor Parody Quality

1. Check your DPO dataset has enough examples (>100 pairs recommended)
2. Try lower beta (0.05) for more aggressive preference learning
3. Train for more epochs (5-10)
4. Use larger model (3B or 7B)

### Dataset Not Loading

```bash
# Login to HuggingFace
huggingface-cli login

# Or set token
export HF_TOKEN="your-token"
```

---

## Cost Estimates

| Setup | Training Time | GPU Cost | Total |
|-------|--------------|----------|-------|
| RTX 3090 + 1.5B | ~2 hours | $0.22/hr | **~$0.50** |
| RTX 3090 + 3B | ~3 hours | $0.22/hr | **~$0.70** |
| RTX 4090 + 3B | ~2 hours | $0.44/hr | **~$0.90** |
| A40 + 7B | ~4 hours | $0.39/hr | **~$1.60** |

**Most cost-effective**: RTX 3090 with Qwen2.5-1.5B for under $1 total.

---

## Automated Training with RunPod API

Instead of manually creating pods, you can automate the entire workflow using the RunPod Python SDK.

### Prerequisites

1. Get your RunPod API key from [runpod.io/console/user/settings](https://www.runpod.io/console/user/settings)
2. Get a HuggingFace token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### Local CLI Usage

```bash
# Install RunPod SDK
pip install runpod

# Set API keys
export RUNPOD_API_KEY="your-runpod-key"
export HF_TOKEN="your-hf-token"

# Launch training (creates pod, runs training, pushes to HF, terminates pod)
python runpod_train.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --dataset patruff/chuckles-dpo \
    --output-repo your-username/parody-1.5b-dpo \
    --gpu rtx3090 \
    --epochs 3

# Launch without waiting (returns pod ID immediately)
python runpod_train.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --dataset patruff/chuckles-dpo \
    --output-repo your-username/parody-1.5b-dpo \
    --no-wait

# Check pod status
python runpod_train.py --status --pod-id abc123xyz

# Terminate a pod
python runpod_train.py --terminate --pod-id abc123xyz

# List available GPU options
python runpod_train.py --list-gpus
```

### GPU Options

| Flag | GPU | Cost/hr | VRAM |
|------|-----|---------|------|
| `rtx3090` | RTX 3090 | ~$0.22 | 24GB |
| `rtx4090` | RTX 4090 | ~$0.44 | 24GB |
| `a40` | NVIDIA A40 | ~$0.39 | 48GB |
| `l40` | NVIDIA L40 | ~$0.49 | 48GB |
| `a100-40` | A100 40GB | ~$1.09 | 40GB |
| `a100-80` | A100 80GB | ~$1.69 | 80GB |

---

## GitHub Actions Automation

The repository includes a GitHub Actions workflow for fully automated training.

### Setup

1. Add these secrets to your GitHub repository (Settings → Secrets → Actions):
   - `RUNPOD_API_KEY`: Your RunPod API key
   - `HF_TOKEN`: Your HuggingFace token with write access

### Running Training via GitHub Actions

1. Go to **Actions** → **Train on RunPod**
2. Click **Run workflow**
3. Fill in the parameters:
   - **Model**: Base model to fine-tune
   - **Dataset**: DPO dataset on HuggingFace
   - **Output Repo**: Where to push the trained model (e.g., `your-username/parody-model`)
   - **GPU**: GPU type (rtx3090 is cheapest)
   - **Epochs**: Training epochs (3 recommended)
   - **Min Pass Rate**: Evaluation threshold (0.7 = 70%)

4. Click **Run workflow**

The workflow will:
1. Launch a RunPod pod with the selected GPU
2. Clone the repo and install dependencies
3. Run DPO training
4. Evaluate the model
5. Push to HuggingFace Hub
6. Terminate the pod
7. Show results in the job summary

### Workflow Parameters

```yaml
inputs:
  model:        # Base model (default: Qwen/Qwen2.5-1.5B-Instruct)
  dataset:      # DPO dataset (default: patruff/chuckles-dpo)
  output_repo:  # HuggingFace repo for output (REQUIRED)
  gpu:          # GPU type (default: rtx3090)
  epochs:       # Training epochs (default: 3)
  batch_size:   # Batch size (default: 2)
  min_pass_rate: # Eval threshold (default: 0.7)
  wait_for_completion: # Wait for training (default: true)
```

### Sample Workflow Run

```
## Training Job Summary

| Parameter | Value |
|-----------|-------|
| Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Dataset | `patruff/chuckles-dpo` |
| Output Repo | [your-username/parody-1.5b](https://huggingface.co/your-username/parody-1.5b) |
| GPU | `rtx3090` |
| Epochs | 3 |
| Pod ID | `abc123xyz` |

✅ **Training completed successfully!**

View model: https://huggingface.co/your-username/parody-1.5b
```

---

## End-to-End Automated Pipeline

Here's the complete automated workflow:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Generate DPO   │───▶│  Train Model    │───▶│  Evaluate &     │
│  Dataset        │    │  on RunPod      │    │  Push to HF     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
       │                       │                       │
       ▼                       ▼                       ▼
  Review CSVs            GitHub Action           Model ready
  via Drive app          triggers RunPod         on HuggingFace
```

### Step-by-Step

1. **Generate parodies for review** (via Drive app or PR workflow)
2. **Review and approve/reject** parodies in CSVs
3. **Process reviews into DPO dataset** (uploads to HuggingFace)
4. **Trigger training workflow** (GitHub Actions → RunPod)
5. **Model is trained, evaluated, and pushed** to HuggingFace
6. **Use the model** for parody generation

All of this can run without any manual SSH or pod management!
