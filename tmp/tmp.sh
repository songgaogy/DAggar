cd /home/dodo/Documents/DAggar/robosuite

TASKS=(
  PandaPickPlaceCan
  PickPlaceMilk
)

for task in "${TASKS[@]}"; do
  echo "[qv_cache_batch] start ${task}"

  MPLCONFIGDIR=/tmp/matplotlib \
  TASK_FILTER="${task}" \
  EXPERT_NUM_TRAJ=20 \
  SUCCESS_NUM_TRAJ=80 \
  FAIL_NUM_TRAJ=80 \
  VALUE_WARMUP_STEPS=20000 \
  FORCE_REBUILD=false \
  bash robosuite/pipeline/scripts/build_awr_qv_cache.sh \
    logging.log_interval=500

  echo "[qv_cache_batch] done ${task}"
done
