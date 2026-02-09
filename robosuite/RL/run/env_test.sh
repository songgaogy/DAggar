set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1

python "${REPO_ROOT}/scripts/env_test.py" \
    --mode both \
    --steps 300 \
    --print-every 10 \
    --reset-on-done