#!/bin/bash
# Runs ON the RunPod pod as the container command (launched by run_pipeline.py
# via docker args -- no SSH needed). Trains the reasoning adapter, optionally
# smoke-tests inference, uploads the run log, then exits so the pod stops.
#
# Expected env (set on the pod by run_pipeline.py):
#   HF_TOKEN       - HF write token (dataset read + adapter push)
#   REPO_URL       - git URL of this repo (default public GitHub URL)
#   GIT_REF        - branch/tag to run (default main)
#   BASE_MODEL     - base model id (ignored in test mode)
#   DATASET_REPO   - HF dataset repo built by build_reasoning_dataset.py
#   ADAPTER_REPO   - HF repo to push the LoRA adapter to
#   EPOCHS         - training epochs
#   TEST_MODE      - "true" for the cheap 0.6B validation run
#   RUN_INFERENCE  - "true" to generate parodies with the adapter
#   INFER_TITLES   - CSV of titles for inference (default: 1995-1999 movies)
#   INFER_LIMIT    - max titles to generate (0 = all; test mode uses 3)
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/patruff/chuck26.git}"
GIT_REF="${GIT_REF:-main}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
DATASET_REPO="${DATASET_REPO:-patruff/chuckles-reasoning-sft}"
ADAPTER_REPO="${ADAPTER_REPO:-patruff/chuckles-reasoning-adapter}"
EPOCHS="${EPOCHS:-3}"
TEST_MODE="${TEST_MODE:-false}"
RUN_INFERENCE="${RUN_INFERENCE:-true}"
INFER_TITLES="${INFER_TITLES:-pipeline/movie_titles_1995_1999.csv}"
if [ "$TEST_MODE" = "true" ]; then
    INFER_LIMIT="${INFER_LIMIT:-3}"
else
    INFER_LIMIT="${INFER_LIMIT:-0}"
fi

LOG=/workspace/run.log
mkdir -p /workspace
exec > >(tee "$LOG") 2>&1

upload_artifacts() {
    # Push the run log (and inference results, if any) to the dataset repo so
    # the orchestrator can retrieve them even after the pod is terminated.
    # A FAILED marker here fail-fasts the orchestrator's poll loop.
    rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "[${RUN_ID:-unknown}] BOOTSTRAP FAILED (exit $rc)"
    fi
    sleep 2  # let the tee process flush the log before uploading
    python -c "
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ['HF_TOKEN'])
api.upload_file(
    path_or_fileobj='$LOG',
    path_in_repo='logs/pod-run-latest.log',
    repo_id='$DATASET_REPO',
    repo_type='dataset',
)
print('run log uploaded')
if os.path.exists('/workspace/inference_results.jsonl'):
    api.upload_file(
        path_or_fileobj='/workspace/inference_results.jsonl',
        path_in_repo='results/inference-latest.jsonl',
        repo_id='$DATASET_REPO',
        repo_type='dataset',
    )
    print('inference results uploaded')
" || echo "WARNING: artifact upload failed"
}
trap upload_artifacts EXIT

echo "=== chucklesPRIME reasoning pipeline pod bootstrap ==="
echo "repo=$REPO_URL ref=$GIT_REF base=$BASE_MODEL test=$TEST_MODE"
nvidia-smi || true

cd /workspace
rm -rf chuck26
git clone --depth 1 --branch "$GIT_REF" "$REPO_URL" chuck26
cd chuck26

echo "=== Installing dependencies ==="
pip install -q -U "transformers>=4.51" peft trl datasets accelerate bitsandbytes huggingface_hub
if [ "$RUN_INFERENCE" = "true" ]; then
    # smolagents + pronouncing are only needed for the tool-scored inference
    # smoke test, not for training.
    pip install -q smolagents pronouncing
    pip install -q -e . --no-deps
fi

TRAIN_FLAGS=(--dataset-repo "$DATASET_REPO" --hf-repo "$ADAPTER_REPO" --epochs "$EPOCHS")
if [ "$TEST_MODE" = "true" ]; then
    TRAIN_FLAGS+=(--test)
else
    TRAIN_FLAGS+=(--base-model "$BASE_MODEL")
fi

echo "=== Training ==="
python pipeline/train_reasoning_sft.py "${TRAIN_FLAGS[@]}"

if [ "$RUN_INFERENCE" = "true" ]; then
    echo "=== Inference: $INFER_TITLES (limit $INFER_LIMIT) ==="
    python pipeline/generate_with_reasoning.py \
        --adapter "$ADAPTER_REPO" \
        --titles "$INFER_TITLES" --limit "$INFER_LIMIT" \
        --output /workspace/inference_results.jsonl \
        || echo "WARNING: inference failed (adapter is already pushed)"
fi

echo "=== Pod work complete ==="
echo "[${RUN_ID:-unknown}] BOOTSTRAP COMPLETE"
