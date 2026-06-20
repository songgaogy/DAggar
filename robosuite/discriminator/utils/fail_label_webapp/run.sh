#!/usr/bin/env bash
# Launch the fail-frame labeling web tool (runs on the REMOTE server).
#
# One-time setup (flask is the only missing dep in `dagger` env; env is
# root-owned so install to user-site with a reachable mirror):
#   /home/dodo/miniconda3/envs/dagger/bin/pip install --user flask \
#       -i https://pypi.tuna.tsinghua.edu.cn/simple
#
# Usage (on the remote server):
#   bash robosuite/discriminator/utils/fail_label_webapp/run.bash
#   PORT=5050 bash .../run.bash          # custom port
#   HOST=0.0.0.0 bash .../run.bash       # expose on all interfaces (LAN access)
#
# ---- Access from your local Mac GUI browser ----
# Option A (recommended, secure): SSH port-forward, then open localhost on Mac.
#   On your Mac:   ssh -N -L 5000:127.0.0.1:5000 <user>@<remote-host>
#   Then browse:   http://localhost:5000
#   (If you opened the repo via Cursor/VS Code Remote-SSH, the port is usually
#    auto-forwarded for you -- just open http://localhost:5000 on the Mac.)
#
# Option B (direct): start with HOST=0.0.0.0 and open http://<remote-ip>:<port>
#   on the Mac. Requires the remote firewall to allow the port.
set -e

PY=/home/dodo/miniconda3/envs/dagger/bin/python
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# repo root = 4 levels up: .../robosuite/discriminator/utils/fail_label_webapp
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"

echo "[run] data root : $DATA_ROOT"
echo "[run] listening : http://$HOST:$PORT"
if [ "$HOST" = "127.0.0.1" ]; then
  echo "[run] remote->Mac: ssh -N -L $PORT:127.0.0.1:$PORT <user>@<remote-host>  (then open http://localhost:$PORT)"
fi

exec "$PY" "$SCRIPT_DIR/server.py" --host "$HOST" --port "$PORT" --data-root "$DATA_ROOT" "$@"
