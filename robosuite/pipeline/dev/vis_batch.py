"""Visualize offline DIPOLE training batches as chunk videos with G / A / failure HUD.

Sanity tool for **checking whether the IQL weight (per-batch G) is correct**: it
rebuilds the exact offline pipeline (:func:`build_offline_pipeline`), draws a few
training batches straight from the mixed replay buffer, and renders each batch as
one short MP4. Every sampled chunk becomes one cell in a grid that plays its
``action_horizon`` observation frames simultaneously, with the sample's
``G = alpha * A - beta * failure``, the TD advantage ``A`` and the nnPU failure
score burned into a HUD. High-G / low-failure chunks should visibly look like
good action segments; low-G / high-failure ones should look off.

The training batch only stores the chunk's *first* frame, so the full chunk
imagery is re-read from ``agent.online_buffer._storage[start : start + H]`` using
the ``start_indices`` that :meth:`DipoleReplayBuffer.sample` reports (buffer holds
uint8 frames already center-cropped to ``image_size`` — no env re-render).
robosuite frames are stored upside-down, so each frame is flipped vertically
before rendering (matching ``eval_dipole._capture_frame``).

Run as a Hydra module (shares the offline config)::

    python -m robosuite.pipeline.dev.vis_batch \\
        env.environment=PickPlaceCereal \\
        runtime.init_checkpoint=.../flow.pt \\
        algorithm.discriminator.checkpoint=.../pu_bce_head.pth \\
        algorithm.q_learning.warmup_ckpt=.../iql_state.pt \\
        +vis.batch_size=6 +vis.num_batches=8 +vis.out_dir=.../vis_batch
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import hydra
import imageio.v2 as imageio
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw, ImageFont

from robosuite.pipeline.algorithms.flow_dagger.replay_buffer import _center_crop_resize
from robosuite.pipeline.offline.train_offline_dipole import build_offline_pipeline

logger = logging.getLogger(__name__)


def _load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", int(size))
    except OSError:
        return ImageFont.load_default()


def _resize_cell(image: np.ndarray, size: int) -> np.ndarray:
    """Resize a HWC uint8 image to (size, size) with a smooth (LANCZOS) filter."""
    pil = Image.fromarray(np.ascontiguousarray(image.astype(np.uint8)))
    if pil.size != (size, size):
        pil = pil.resize((size, size), Image.LANCZOS)
    return np.asarray(pil, dtype=np.uint8)


def _overlay_cell(
    image: np.ndarray,
    *,
    sample_idx: int,
    g: float,
    adv: float,
    fail: float,
    font: ImageFont.ImageFont,
    line_h: int,
    bar_h: int,
) -> np.ndarray:
    """Burn a translucent HUD (sample idx + G / A / failure) onto one cell."""
    pil = Image.fromarray(np.ascontiguousarray(image)).convert("RGBA")
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width = pil.size[0]
    pad = max(4, line_h // 4)
    draw.rectangle([0, 0, width, bar_h], fill=(0, 0, 0, 170))
    draw.text((pad, pad), f"s{int(sample_idx)}", fill=(255, 255, 255, 255), font=font)
    draw.text((pad, pad + line_h), f"G={float(g):+.3f}  A={float(adv):+.3f}", fill=(120, 220, 255, 255), font=font)
    draw.text((pad, pad + 2 * line_h), f"fail={float(fail):+.3f}", fill=(255, 180, 120, 255), font=font)
    return np.asarray(Image.alpha_composite(pil, overlay).convert("RGB"), dtype=np.uint8)


def _grid_dims(n: int) -> tuple[int, int]:
    """(rows, cols) laid out wider-than-tall, e.g. 6 -> (2, 3), 8 -> (2, 4)."""
    rows = max(1, int(math.floor(math.sqrt(n))))
    cols = int(math.ceil(n / rows))
    return rows, cols


def _tile_grid(cells: list[np.ndarray], rows: int, cols: int) -> np.ndarray:
    """Tile rows*cols equal-size cells (already padded) into one image."""
    row_imgs = [np.concatenate(cells[r * cols : (r + 1) * cols], axis=1) for r in range(rows)]
    return np.concatenate(row_imgs, axis=0)


def _add_header(grid: np.ndarray, text: str, font: ImageFont.ImageFont, height: int) -> np.ndarray:
    """Prepend a black navigation bar with `text` above the grid."""
    bar = np.zeros((int(height), grid.shape[1], 3), dtype=np.uint8)
    pil = Image.fromarray(bar)
    ImageDraw.Draw(pil).text((6, max(2, height // 5)), text, fill=(255, 255, 255), font=font)
    return np.concatenate([np.asarray(pil, dtype=np.uint8), grid], axis=0)


def _pad_even(frame: np.ndarray) -> np.ndarray:
    """Pad bottom/right by 1px so H and W are even (libx264 yuv420p needs even dims)."""
    pad_h = frame.shape[0] % 2
    pad_w = frame.shape[1] % 2
    if pad_h or pad_w:
        frame = np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
    return frame


def _write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        path,
        format="FFMPEG",
        mode="I",
        fps=int(fps),
        codec="libx264",
        ffmpeg_params=["-movflags", "+faststart"],
        macro_block_size=1,
    ) as writer:
        for frame in frames:
            writer.append_data(np.ascontiguousarray(_pad_even(frame)))


@hydra.main(version_base="1.2", config_path="../config", config_name="offline")
def main(cfg: DictConfig) -> None:
    torch.set_grad_enabled(False)

    # ------------------------------------------------------------------ #
    # vis-specific params (passed as `+vis.*` hydra additions).           #
    # ------------------------------------------------------------------ #
    task_name = str(cfg.env.environment)
    vis_bs = int(OmegaConf.select(cfg, "vis.batch_size", default=6))
    num_batches = int(OmegaConf.select(cfg, "vis.num_batches", default=8))
    fps = int(OmegaConf.select(cfg, "vis.fps", default=4))
    cell_px = int(OmegaConf.select(cfg, "vis.cell_px", default=384))
    vis_seed = OmegaConf.select(cfg, "vis.seed", default=0)
    out_dir_raw = OmegaConf.select(
        cfg, "vis.out_dir", default=f"outputs/dipole_offline/{task_name}/dev/vis_batch"
    )
    out_dir = Path(to_absolute_path(str(out_dir_raw)))
    out_dir.mkdir(parents=True, exist_ok=True)

    # HUD geometry scales with the (larger) cell resolution so text stays legible.
    font_size = max(11, cell_px // 22)
    font = _load_font(font_size)
    line_h = font_size + 3
    bar_h = line_h * 3 + 2 * max(4, line_h // 4)
    header_px = font_size + 12
    header_font = _load_font(max(12, cell_px // 24))

    # ------------------------------------------------------------------ #
    # Rebuild the exact offline pipeline (frozen critics + G provider).   #
    # ------------------------------------------------------------------ #
    pipe = build_offline_pipeline(cfg)
    agent = pipe.agent
    buffer = agent.online_buffer
    H = int(pipe.iql_cfg.action_horizon)
    image_size = int(buffer.image_size)
    main_cam = str(pipe.policy_camera_names[0])
    alpha = float(pipe.provider.alpha)
    beta = float(pipe.provider.beta)

    print(
        f"[vis_batch] task={task_name} main_cam={main_cam} H={H} image_size={image_size} "
        f"cell_px={cell_px} alpha={alpha} beta={beta} vis_bs={vis_bs} "
        f"num_batches={num_batches} fps={fps}"
    )
    print(f"[vis_batch] out_dir={out_dir}")

    if vis_seed is not None:
        np.random.seed(int(vis_seed))

    # Draw clean (un-augmented) batches for visualization.
    sample_kwargs = {**pipe.sample_kwargs, "augment": False}
    rows, cols = _grid_dims(vis_bs)
    black_cell = np.zeros((cell_px, cell_px, 3), dtype=np.uint8)

    for b in range(num_batches):
        batch = buffer.sample(vis_bs, **sample_kwargs)
        starts = [int(s) for s in batch.metadata["start_indices"]]
        is_int = batch.is_intervention.reshape(-1).tolist()

        # G / A / failure from the precomputed tables (same values the trainer sees).
        rows_idx = torch.tensor([pipe.start_to_row[s] for s in starts], dtype=torch.long)
        adv = pipe.advantage_raw.index_select(0, rows_idx).reshape(-1)
        fail = pipe.failure_raw.index_select(0, rows_idx).reshape(-1)
        g = (alpha * adv - beta * fail).reshape(-1)
        adv_l, fail_l, g_l = adv.tolist(), fail.tolist(), g.tolist()

        # Re-read each chunk's H main-camera frames from buffer storage.
        # robosuite frames are stored upside-down -> flip vertically for display.
        with buffer._lock:  # noqa: SLF001 — read a static buffer under its own lock
            sequences = [list(buffer._storage[s : s + H]) for s in starts]  # noqa: SLF001
        clips: list[list[np.ndarray]] = []
        for seq in sequences:
            frames = [
                np.flipud(
                    _center_crop_resize(np.asarray(tr.obs[main_cam], dtype=np.uint8), image_size)
                )
                for tr in seq
            ]
            clips.append(frames)

        # Build one video frame per chunk timestep (all samples play in sync).
        video_frames: list[np.ndarray] = []
        for t in range(H):
            cells: list[np.ndarray] = []
            for i in range(rows * cols):
                if i >= vis_bs:
                    cells.append(black_cell)
                    continue
                cell = _resize_cell(clips[i][t], cell_px)
                cell = _overlay_cell(
                    cell,
                    sample_idx=i,
                    g=g_l[i],
                    adv=adv_l[i],
                    fail=fail_l[i],
                    font=font,
                    line_h=line_h,
                    bar_h=bar_h,
                )
                cells.append(cell)
            grid = _tile_grid(cells, rows, cols)
            grid = _add_header(grid, f"batch {b}  frame {t + 1}/{H}", header_font, header_px)
            video_frames.append(grid)

        video_path = out_dir / f"batch_{b:02d}.mp4"
        _write_video(video_path, video_frames, fps=fps)
        n_pre = int(sum(1 for v in is_int if bool(v)))
        print(
            f"[vis_batch][batch {b:02d}] -> {video_path.name}  "
            f"pretrain={n_pre}/{vis_bs}  "
            f"G[min/mean/max]={min(g_l):+.3f}/{float(np.mean(g_l)):+.3f}/{max(g_l):+.3f}  "
            f"A_mean={float(np.mean(adv_l)):+.3f}  fail_mean={float(np.mean(fail_l)):+.3f}"
        )

    print(f"[vis_batch] done. wrote {num_batches} videos to {out_dir}")


if __name__ == "__main__":
    main()
