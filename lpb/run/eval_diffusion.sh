export CUDA_VISIBLE_DEVICES=1
export LPB_RENDER_GPU_IDS=1
export LPB_VECTOR_ENV_MODE=auto
export LPB_VECTOR_ENV_CONTEXT=spawn 

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-$MUJOCO_GL}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH"
  exit 1
fi
eval "$(conda shell.bash hook)"
conda activate lpb

python /home/dodo/Documents/DAggar/robosuite/lpb/eval_base_diffusion_policy.py \
  --policy-checkpoint /home/dodo/Documents/DAggar/robosuite/lpb/data/outputs/2026.03.09/11.53.10_train_diffusion_unet_hybrid_transport_image/checkpoints/200.ckpt \
  --output-dir /home/dodo/Documents/DAggar/robosuite/lpb/data/eval/base_transport/ep200 \
  --device cuda:0 \
  --n-test 50 \
  --test-start-seed 100000 \
  --success-reward-threshold 0.5
