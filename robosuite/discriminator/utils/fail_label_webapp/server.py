"""Flask backend for the fail-frame labeling web tool.

Reads rollout HDF5 files under data/<task>/fail_rollout/, serves frames for
viewing/labeling, and exports labeled HDF5 (full parity with
data/utils/fail_labeled_train) to data/<task>/fail_rollout-labeled/.
"""
import argparse
import io
import json
import logging
import os
import re
import threading
from collections import OrderedDict

import cv2
import h5py
import numpy as np
from flask import Flask, Response, abort, jsonify, request, send_from_directory

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HERE, "static")
IN_SUBDIR = "fail_rollout-val"


def _find_data_root():
    """Walk up from this file to locate the repo's `data/` dir.

    Location-independent: works no matter where this webapp folder lives, as
    long as a sibling/ancestor `data/<task>/fail_rollout` layout exists.
    Falls back to <repo>/data assuming repo == 4 levels up from HERE.
    """
    cur = HERE
    for _ in range(8):
        cand = os.path.join(cur, "data")
        if os.path.isdir(cand):
            # prefer a data dir that actually has the expected layout
            try:
                for name in os.listdir(cand):
                    if os.path.isdir(os.path.join(cand, name, IN_SUBDIR)):
                        return os.path.abspath(cand)
            except OSError:
                pass
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    # fallback: nearest ancestor `data` dir if any, else repo-relative guess
    cur = HERE
    for _ in range(8):
        cand = os.path.join(cur, "data")
        if os.path.isdir(cand):
            return os.path.abspath(cand)
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "data"))


DATA_ROOT = _find_data_root()
OUT_SUBDIR = "fail_rollout-val-labeled"
DEMOS_GROUP = "demos"
SEGMENT_FORMAT_STR = (
    'json list of {"start","end","mode"}; 0-based inclusive frame indices; '
    "stored values are clamped to valid range. \"mode\" is a row-level label "
    "and may be an empty string."
)
SEGMENT_INDEX_MEANING = (
    "per-frame index into failure_segments_json list (-1 = not in any segment)"
)

app = Flask(__name__, static_folder=None)

# ---------------------------------------------------------------------------
# HDF5 handle LRU cache (avoid reopening per frame request)
# ---------------------------------------------------------------------------
_H5_CACHE = OrderedDict()  # path -> h5py.File
_H5_CACHE_MAX = 4
_H5_LOCK = threading.Lock()


def get_h5(path):
    """Return a cached read-only h5py.File handle for `path`."""
    with _H5_LOCK:
        if path in _H5_CACHE:
            _H5_CACHE.move_to_end(path)
            return _H5_CACHE[path]
        if not os.path.isfile(path):
            abort(404, f"file not found: {path}")
        f = h5py.File(path, "r")
        _H5_CACHE[path] = f
        _H5_CACHE.move_to_end(path)
        while len(_H5_CACHE) > _H5_CACHE_MAX:
            _, old = _H5_CACHE.popitem(last=False)
            try:
                old.close()
            except Exception:
                pass
        return f


# ---------------------------------------------------------------------------
# Path helpers (with basic traversal protection)
# ---------------------------------------------------------------------------
def _safe_name(name):
    if not name or os.sep in name or (os.altsep and os.altsep in name) or name in (".", ".."):
        abort(400, f"invalid name: {name}")
    return name


def task_in_dir(task):
    return os.path.join(DATA_ROOT, _safe_name(task), IN_SUBDIR)


def task_out_dir(task):
    return os.path.join(DATA_ROOT, _safe_name(task), OUT_SUBDIR)


def in_file_path(task, file):
    return os.path.join(task_in_dir(task), _safe_name(file))


def sidecar_path(task, file):
    stem = os.path.splitext(_safe_name(file))[0]
    return os.path.join(task_out_dir(task), f".labels__{stem}.json")


def labeled_output_filename(src_file, n_labeled):
    """Map source HDF5 name to output name with actual labeled demo count.

    e.g. fail_rollout_20260420_150.hdf5 + 30 labeled -> fail_rollout_20260420_30.hdf5
    """
    safe = _safe_name(src_file)
    stem, ext = os.path.splitext(safe)
    if not ext:
        ext = ".hdf5"
    m = re.match(r"^(.+)_(\d+)$", stem)
    if m:
        return f"{m.group(1)}_{n_labeled}{ext}"
    return f"{stem}_{n_labeled}{ext}"


# ---------------------------------------------------------------------------
# Sidecar (progress) read/write
# ---------------------------------------------------------------------------
_SIDECAR_LOCK = threading.Lock()


def load_sidecar(task, file):
    p = sidecar_path(task, file)
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, "r") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_sidecar(task, file, data):
    p = sidecar_path(task, file)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with _SIDECAR_LOCK:
        tmp = p + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, p)


# ---------------------------------------------------------------------------
# Demo / camera helpers
# ---------------------------------------------------------------------------
def get_demos_group(f):
    if DEMOS_GROUP in f:
        return f[DEMOS_GROUP]
    abort(500, f"'{DEMOS_GROUP}' group not found in file")


def list_cameras(demo_grp):
    obs = demo_grp.get("observations")
    if obs is None:
        return []
    cams = [k for k in obs.keys() if "images" in obs[k]]
    return sorted(cams)


def demo_length(demo_grp):
    if "length" in demo_grp.attrs:
        return int(demo_grp.attrs["length"])
    obs = demo_grp.get("observations")
    if obs is not None:
        for k in obs.keys():
            if "images" in obs[k]:
                return int(obs[k]["images"].shape[0])
    if "actions" in demo_grp:
        return int(demo_grp["actions"].shape[0])
    return 0


# ---------------------------------------------------------------------------
# Routes: static
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/static/<path:fname>")
def static_files(fname):
    return send_from_directory(STATIC_DIR, fname)


# ---------------------------------------------------------------------------
# Routes: API
# ---------------------------------------------------------------------------
@app.route("/api/tasks")
def api_tasks():
    tasks = []
    if os.path.isdir(DATA_ROOT):
        for name in sorted(os.listdir(DATA_ROOT)):
            in_dir = os.path.join(DATA_ROOT, name, IN_SUBDIR)
            if os.path.isdir(in_dir) and any(
                fn.endswith(".hdf5") for fn in os.listdir(in_dir)
            ):
                tasks.append(name)
    return jsonify({"tasks": tasks})


@app.route("/api/files")
def api_files():
    task = request.args.get("task", "")
    in_dir = task_in_dir(task)
    files = []
    if os.path.isdir(in_dir):
        files = sorted(fn for fn in os.listdir(in_dir) if fn.endswith(".hdf5"))
    return jsonify({"files": files})


@app.route("/api/demos")
def api_demos():
    task = request.args.get("task", "")
    file = request.args.get("file", "")
    path = in_file_path(task, file)
    f = get_h5(path)
    g = get_demos_group(f)
    sidecar = load_sidecar(task, file)
    demos = []
    cameras = None
    for key in sorted(g.keys()):
        d = g[key]
        if cameras is None:
            cameras = list_cameras(d)
        label = sidecar.get(key)
        demos.append(
            {
                "demo": key,
                "length": demo_length(d),
                "labeled": label is not None,
                "label": label,
            }
        )
    n_labeled = sum(1 for d in demos if d["labeled"])
    return jsonify(
        {
            "demos": demos,
            "cameras": cameras or [],
            "default_camera": "agentview" if cameras and "agentview" in cameras else (cameras[0] if cameras else None),
            "n_total": len(demos),
            "n_labeled": n_labeled,
        }
    )


@app.route("/api/frame")
def api_frame():
    task = request.args.get("task", "")
    file = request.args.get("file", "")
    demo = request.args.get("demo", "")
    cam = request.args.get("cam", "")
    try:
        idx = int(request.args.get("idx", "0"))
    except ValueError:
        abort(400, "idx must be int")
    quality = int(request.args.get("q", "80"))

    path = in_file_path(task, file)
    f = get_h5(path)
    g = get_demos_group(f)
    if demo not in g:
        abort(404, f"demo not found: {demo}")
    d = g[demo]
    obs = d.get("observations")
    if obs is None or cam not in obs or "images" not in obs[cam]:
        abort(404, f"camera not found: {cam}")
    ds = obs[cam]["images"]
    n = ds.shape[0]
    if idx < 0:
        idx = 0
    if idx >= n:
        idx = n - 1
    arr = np.asarray(ds[idx])  # (H,W,3) RGB uint8
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    bgr = cv2.flip(bgr, 0)  # vertical flip for display
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        abort(500, "jpeg encode failed")
    resp = Response(buf.tobytes(), mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "public, max-age=3600"
    return resp


def _clamp_segment(start, end, length):
    """Normalize one (start, end) to 0-based inclusive valid indices."""
    if length <= 0:
        return 0, 0
    last = length - 1
    if start is None:
        start = 0
    if end is None:
        end = last
    start = int(start)
    end = int(end)
    # CSV convention: end == length means "through last frame"
    if end >= length:
        end = last
    if start >= length:
        start = last
    if start < 0:
        start = 0
    if end < 0:
        end = 0
    if start > end:
        start, end = end, start
    return start, end


@app.route("/api/label", methods=["POST"])
def api_label():
    body = request.get_json(force=True, silent=True) or {}
    task = body.get("task", "")
    file = body.get("file", "")
    demo = body.get("demo", "")
    if not demo:
        abort(400, "demo required")
    start = body.get("start")
    end = body.get("end")
    mode = body.get("mode", "") or ""

    # validate against actual demo length
    path = in_file_path(task, file)
    f = get_h5(path)
    g = get_demos_group(f)
    if demo not in g:
        abort(404, f"demo not found: {demo}")
    length = demo_length(g[demo])
    s, e = _clamp_segment(start, end, length)

    sidecar = load_sidecar(task, file)
    sidecar[demo] = {"start": s, "end": e, "mode": str(mode)}
    save_sidecar(task, file, sidecar)
    n_labeled = len(sidecar)
    return jsonify({"ok": True, "demo": demo, "label": sidecar[demo], "n_labeled": n_labeled})


@app.route("/api/delete_label", methods=["POST"])
def api_delete_label():
    body = request.get_json(force=True, silent=True) or {}
    task = body.get("task", "")
    file = body.get("file", "")
    demo = body.get("demo", "")
    sidecar = load_sidecar(task, file)
    if demo in sidecar:
        del sidecar[demo]
        save_sidecar(task, file, sidecar)
    return jsonify({"ok": True, "demo": demo, "n_labeled": len(sidecar)})


@app.route("/api/export", methods=["POST"])
def api_export():
    body = request.get_json(force=True, silent=True) or {}
    task = body.get("task", "")
    file = body.get("file", "")
    src_path = in_file_path(task, file)
    sidecar = load_sidecar(task, file)
    labeled = {k: v for k, v in sidecar.items() if v is not None}
    if not labeled:
        return jsonify({"ok": False, "error": "no labeled demos to export"}), 400

    out_dir = task_out_dir(task)
    os.makedirs(out_dir, exist_ok=True)

    # Only export demos present in both sidecar and source HDF5.
    with h5py.File(src_path, "r") as src:
        src_demos = src[DEMOS_GROUP]
        demos_to_export = sorted(d for d in labeled.keys() if d in src_demos)
    if not demos_to_export:
        return jsonify({"ok": False, "error": "no labeled demos found in source HDF5"}), 400

    out_name = labeled_output_filename(file, len(demos_to_export))
    out_path = os.path.join(out_dir, out_name)

    n_written = 0
    # Read source in its own handle (do not use cached read handle for the long copy)
    with h5py.File(src_path, "r") as src, h5py.File(out_path, "w") as out:
        meta = out.create_group("meta")
        meta.attrs["source_hdf5_path"] = os.path.abspath(src_path)
        meta.attrs["source_labels_json"] = os.path.abspath(sidecar_path(task, file))
        meta.attrs["failure_segment_format"] = SEGMENT_FORMAT_STR
        meta.attrs["n_demos"] = len(demos_to_export)
        out_demos = out.create_group(DEMOS_GROUP)
        src_demos = src[DEMOS_GROUP]

        for demo in demos_to_export:
            lab = labeled[demo]
            d_src = src_demos[demo]
            length = demo_length(d_src)
            s, e = _clamp_segment(lab.get("start"), lab.get("end"), length)
            mode = str(lab.get("mode", "") or "")

            # full copy of original demo content (datasets + attrs)
            src.copy(d_src, out_demos, name=demo)
            d_out = out_demos[demo]

            d_out.attrs["failure_segments_json"] = json.dumps(
                [{"start": s, "end": e, "mode": mode}]
            )

            mask = np.zeros((length,), dtype=np.uint8)
            seg_idx = np.full((length,), -1, dtype=np.int32)
            mask[s : e + 1] = 1
            seg_idx[s : e + 1] = 0

            ann = d_out.create_group("annotations")
            ann.create_dataset("failure_frame_mask", data=mask, compression="gzip")
            ann.create_dataset(
                "failure_segment_index", data=seg_idx, compression="gzip"
            )
            ann.attrs["failure_segment_index_meaning"] = SEGMENT_INDEX_MEANING
            n_written += 1

    return jsonify(
        {
            "ok": True,
            "output_path": os.path.abspath(out_path),
            "output_filename": out_name,
            "n_written": n_written,
        }
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _configure_logging():
    """Suppress per-request access logs; keep Flask startup banner."""

    class _AccessLogFilter(logging.Filter):
        def filter(self, record):
            msg = record.getMessage()
            # werkzeug access log: '"GET /api/frame HTTP/1.1" 200 -'
            if '"GET ' in msg or '"POST ' in msg or '"PUT ' in msg or '"DELETE ' in msg:
                return False
            return True

    logging.getLogger("werkzeug").addFilter(_AccessLogFilter())


def main():
    parser = argparse.ArgumentParser(description="fail-frame labeling web tool")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--data-root", default=None,
                        help="override data root (default: <repo>/data)")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    _configure_logging()

    global DATA_ROOT
    if args.data_root:
        DATA_ROOT = os.path.abspath(args.data_root)
    if not os.path.isdir(DATA_ROOT):
        logging.warning("DATA_ROOT does not exist: %s", DATA_ROOT)

    print(f"[fail-label] DATA_ROOT = {DATA_ROOT}")
    print(f"[fail-label] open http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
