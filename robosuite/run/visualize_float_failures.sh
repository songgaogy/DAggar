set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

EXPERT_DIR="${EXPERT_DIR:-${ROOT_DIR}/data/PandaLift/expert}"
FAIL_DIR="${FAIL_DIR:-${ROOT_DIR}/data/PandaLift/fail_rollout}"
CAMERA_NAME="${CAMERA_NAME:-agentview}"
POLICY_CKPT="${POLICY_CKPT:-${ROOT_DIR}/checkpoints/PandaLift/flow/BC_warmup/flow_policy_ep0040_20260303_135632.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/checkpoints/PandaLift/discriminator/float/vis}"

python "${ROOT_DIR}/robosuite/discriminator/utils/visualize.py" \
  --expert-dir "${EXPERT_DIR}" \
  --fail-dir "${FAIL_DIR}" \
  --camera-name "${CAMERA_NAME}" \
  --policy-ckpt "${POLICY_CKPT}" \
  --policy-device cuda \
  --image-size 128 \
  --latent-batch-size 128 \
  --ta 8 \
  --to 2 \
  --sinkhorn-reg 0.05 \
  --max-iter 200 \
  --tol 1e-5 \
  --num-expert-candidates 100 \
  --delta 10 \
  --delta-step 1 \
  --adaptive-delta-on-fail-train \
  --fail-tail-ratio 0.2 \
  --num-eval-fail 5 \
  --fps 20 \
  --seed 42 \
  --flip-vertical \
  --progress \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
