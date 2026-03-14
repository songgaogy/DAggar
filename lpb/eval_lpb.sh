#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export CUDA_VISIBLE_DEVICES=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM="${MUJOCO_GL}"
export LPB_RENDER_GPU_IDS="${LPB_RENDER_GPU_IDS:-$CUDA_VISIBLE_DEVICES}"
export LPB_VECTOR_ENV_MODE="${LPB_VECTOR_ENV_MODE:-auto}"
export LPB_VECTOR_ENV_CONTEXT="${LPB_VECTOR_ENV_CONTEXT:-spawn}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH"
  exit 1
fi
eval "$(conda shell.bash hook)"
conda activate lpb

DEFAULT_POLICY_CKPT="/home/dodo/Documents/DAggar/robosuite/lpb/data/outputs/2026.03.09/11.53.10_train_diffusion_unet_hybrid_transport_image/checkpoints/20.ckpt"
DEFAULT_DYN_CKPT="/home/dodo/Documents/DAggar/robosuite/lpb/data/outputs/2026.03.12/11.02.54_transport/checkpoints/model_200.pth"

CONFIG_NAME="eval_transport"
POLICY_CHECKPOINT="${POLICY_CHECKPOINT:-$DEFAULT_POLICY_CKPT}"
DYNAMICS_MODEL_CHECKPOINT="${DYNAMICS_MODEL_CHECKPOINT:-$DEFAULT_DYN_CKPT}"
DEMO_DATASET_PATH="${DEMO_DATASET_PATH:-$REPO_ROOT/data/lpb/transport/transport_rollout_and_demo.hdf5}"
DATASET_PATH="${DATASET_PATH:-$DEMO_DATASET_PATH}"

N_TEST=50
N_TEST_VIS=1
TEST_START_SEED=100000
N_ACTION_STEPS=15
GUIDANCE_START_TIMESTEP=10
GUIDANCE_SCALE=0.2
THRESHOLD=2.8
SUCCESS_REWARD_THRESHOLD=0.5
DEVICE="cuda:0"

DEMO_BATCH_SIZE=32
DEMO_LOADER_WORKERS=8
DEMO_SUBSAMPLE_STRIDE=1
DEMO_MAX_SAMPLES=null
NN_CHUNK_SIZE=1024

OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/data/eval/lpb_transport/$(date +%Y.%m.%d_%H.%M.%S)}"
FORCE_OVERWRITE="${FORCE_OVERWRITE:-0}"
BASE_RESULTS_PATH="${BASE_RESULTS_PATH:-}"

if [ -z "$POLICY_CHECKPOINT" ]; then
  echo "policy checkpoint is required. Set POLICY_CHECKPOINT=/path/to/base_policy.ckpt"
  exit 1
fi
if [ -z "$DYNAMICS_MODEL_CHECKPOINT" ]; then
  echo "dynamics checkpoint is required. Set DYNAMICS_MODEL_CHECKPOINT=/path/to/model_XX.pth"
  exit 1
fi
if [ ! -f "$POLICY_CHECKPOINT" ]; then
  echo "policy checkpoint file not found: $POLICY_CHECKPOINT"
  exit 1
fi
if [ ! -f "$DYNAMICS_MODEL_CHECKPOINT" ]; then
  echo "dynamics checkpoint file not found: $DYNAMICS_MODEL_CHECKPOINT"
  exit 1
fi
if [ ! -f "$DEMO_DATASET_PATH" ]; then
  echo "demo dataset file not found: $DEMO_DATASET_PATH"
  exit 1
fi
if [ ! -f "$DATASET_PATH" ]; then
  echo "dataset file not found: $DATASET_PATH"
  exit 1
fi

if command -v python >/dev/null 2>&1; then
  DEMO_ENV_NAME="$(python - <<'PY' "$DEMO_DATASET_PATH" | tail -n 1
import sys
import robomimic.utils.file_utils as F
meta = F.get_env_metadata_from_dataset(dataset_path=sys.argv[1])
print(meta.get('env_name', ''))
PY
)"
  DATA_ENV_NAME="$(python - <<'PY' "$DATASET_PATH" | tail -n 1
import sys
import robomimic.utils.file_utils as F
meta = F.get_env_metadata_from_dataset(dataset_path=sys.argv[1])
print(meta.get('env_name', ''))
PY
)"
  if [ "$CONFIG_NAME" = "eval_transport" ]; then
    if [[ "$DEMO_ENV_NAME" != *Transport* ]]; then
      echo "demo dataset env_name is '$DEMO_ENV_NAME' (expected Transport)."
      echo "DEMO_DATASET_PATH=$DEMO_DATASET_PATH"
      exit 1
    fi
    if [[ "$DATA_ENV_NAME" != *Transport* ]]; then
      echo "dataset env_name is '$DATA_ENV_NAME' (expected Transport)."
      echo "DATASET_PATH=$DATASET_PATH"
      exit 1
    fi
  fi
fi

if [ -e "$OUTPUT_DIR" ] && [ "$FORCE_OVERWRITE" = "1" ]; then
  rm -rf "$OUTPUT_DIR"
fi

echo "=== LPB Test-Time Steering Evaluation ($CONFIG_NAME) ==="
echo "POLICY_CHECKPOINT=$POLICY_CHECKPOINT"
echo "DYNAMICS_MODEL_CHECKPOINT=$DYNAMICS_MODEL_CHECKPOINT"
echo "DEMO_DATASET_PATH=$DEMO_DATASET_PATH"
echo "DATASET_PATH=$DATASET_PATH"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "N_TEST=$N_TEST N_TEST_VIS=$N_TEST_VIS TEST_START_SEED=$TEST_START_SEED"
echo "GUIDANCE_SCALE=$GUIDANCE_SCALE THRESHOLD=$THRESHOLD"
echo "DEMO_BATCH_SIZE=$DEMO_BATCH_SIZE DEMO_SUBSAMPLE_STRIDE=$DEMO_SUBSAMPLE_STRIDE DEMO_MAX_SAMPLES=$DEMO_MAX_SAMPLES"

cd "$SCRIPT_DIR"
python -X faulthandler eval_test_time_optimization.py \
  --config-name="$CONFIG_NAME" \
  policy_checkpoint="$POLICY_CHECKPOINT" \
  dynamics_model_checkpoint="$DYNAMICS_MODEL_CHECKPOINT" \
  demo_dataset_path="$DEMO_DATASET_PATH" \
  ++dataset_path="$DATASET_PATH" \
  n_test="$N_TEST" \
  n_test_vis="$N_TEST_VIS" \
  test_start_seed="$TEST_START_SEED" \
  n_action_steps="$N_ACTION_STEPS" \
  guidance_start_timestep="$GUIDANCE_START_TIMESTEP" \
  guidance_scale="$GUIDANCE_SCALE" \
  threshold="$THRESHOLD" \
  demo_batch_size="$DEMO_BATCH_SIZE" \
  demo_loader_workers="$DEMO_LOADER_WORKERS" \
  demo_subsample_stride="$DEMO_SUBSAMPLE_STRIDE" \
  demo_max_samples="$DEMO_MAX_SAMPLES" \
  nn_chunk_size="$NN_CHUNK_SIZE" \
  output_dir="$OUTPUT_DIR" \
  device="$DEVICE"

SUMMARY_ARGS=(
  --eval-results "$OUTPUT_DIR/eval_results.json"
  --output "$OUTPUT_DIR/eval_lpb_summary.json"
  --success-reward-threshold "$SUCCESS_REWARD_THRESHOLD"
)

if [ -n "$BASE_RESULTS_PATH" ]; then
  SUMMARY_ARGS+=(--base-results "$BASE_RESULTS_PATH")
fi

python summarize_lpb_eval.py "${SUMMARY_ARGS[@]}"
echo "Saved final LPB summary to: $OUTPUT_DIR/eval_lpb_summary.json"
