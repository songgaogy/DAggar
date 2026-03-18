set -euo pipefail

ROOT_DIR="/home/dodo/Documents/DAggar/robosuite"
export CUDA_VISIBLE_DEVICES=0

EXPERT_DIR="/home/dodo/Documents/DAggar/robosuite/data/PandaPickPlaceCan/expert_recover"
FAIL_DIR="/home/dodo/Documents/DAggar/robosuite/data/PandaPickPlaceCan/fail_rollout"
CAMERA_NAME="agentview"
VIS_CAMERA_NAME="agentview"
POLICY_CKPT="/home/dodo/Documents/DAggar/robosuite/checkpoints/PickPlaceCan/flow_unet-10/flow_unet_ep0400_20260318_020356.pt"
OUTPUT_DIR="/home/dodo/Documents/DAggar/robosuite/checkpoints/PickPlaceCan/discriminator/float/vis"
NUM_VIS=5

TOTAL_STEPS=3
CURRENT_STEP=0

print_step() {
  CURRENT_STEP=$((CURRENT_STEP + 1))
  echo "[${CURRENT_STEP}/${TOTAL_STEPS}] $1"
}

print_step "Preparing FLOAT failure visualization run"
echo "expert_dir=${EXPERT_DIR}"
echo "fail_dir=${FAIL_DIR}"
echo "policy_ckpt=${POLICY_CKPT}"
echo "num_vis=${NUM_VIS}"

print_step "Launching visualization with tqdm progress bars"

python -m robosuite.discriminator.float.utils.visualize \
  --expert-dir "${EXPERT_DIR}" \
  --fail-dir "${FAIL_DIR}" \
  --policy-type flow_unet \
  --camera-name "${CAMERA_NAME}" \
  --vis-camera-name "${VIS_CAMERA_NAME}" \
  --policy-ckpt "${POLICY_CKPT}" \
  --policy-device cuda \
  --ot-device cuda \
  --image-size -1 \
  --latent-batch-size 128 \
  --sinkhorn-reg 0.05 \
  --max-iter 100 \
  --tol 1e-5 \
  --num-expert-candidates 50 \
  --delta 10 \
  --delta-step 1 \
  --max-calibration-experts 100 \
  --fail-tail-ratio 0.2 \
  --num-vis $NUM_VIS \
  --fps 20 \
  --seed 42 \
  --flip-vertical \
  --progress \
  --output-dir "${OUTPUT_DIR}" \
  "$@"

print_step "Visualization finished"
