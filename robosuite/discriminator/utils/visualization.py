from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def fix_robosuite_frame_orientation(frame_rgb: np.ndarray, flip_vertical: bool) -> np.ndarray:
    if not bool(flip_vertical):
        return np.asarray(frame_rgb, dtype=np.uint8)
    return np.flipud(np.asarray(frame_rgb, dtype=np.uint8))


def select_camera_frame(
    images_hwc: np.ndarray,
    frame_id: int,
    camera_names: Sequence[str],
    vis_camera_name: Optional[str],
) -> np.ndarray:
    images = np.asarray(images_hwc)
    if images.ndim != 5:
        raise ValueError(f"Expected images shape (T,V,H,W,C), got {images.shape}")
    names = [str(name) for name in camera_names]
    if vis_camera_name and str(vis_camera_name) in names:
        view_idx = names.index(str(vis_camera_name))
    else:
        view_idx = 0
    return images[int(frame_id), view_idx]


def map_step_values_to_frames(
    step_values: np.ndarray,
    num_frames: int,
    tail_fill: Optional[float] = None,
) -> np.ndarray:
    steps = np.asarray(step_values).reshape(-1)
    if int(num_frames) <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if steps.size == 0:
        fill_value = 0.0 if tail_fill is None else float(tail_fill)
        return np.full(int(num_frames), fill_value, dtype=np.float64)

    out = np.full(int(num_frames), float(steps[-1] if tail_fill is None else tail_fill), dtype=np.float64)
    valid = min(int(num_frames), int(steps.size))
    out[:valid] = steps[:valid]
    return out


def draw_detection_overlay(
    frame_rgb: np.ndarray,
    *,
    frame_id: int,
    pred_fail_flag: bool,
    gt_fail_flag: bool,
    aggregate_score: float,
    threshold: float,
    detector_name: str,
    extra_text: Optional[str] = None,
    border_thickness: int = 5,
    banner_font_scale: float = 1.0,
    banner_thickness: int = 2,
    footer_font_scale: float = 0.65,
    footer_thickness: int = 2,
) -> np.ndarray:
    frame = np.asarray(frame_rgb, dtype=np.uint8).copy()
    if cv2 is None:
        if pred_fail_flag:
            frame[:8, :, 0] = 255
            frame[-8:, :, 0] = 255
            frame[:, :8, 0] = 255
            frame[:, -8:, 0] = 255
        return frame

    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]

    if pred_fail_flag:
        cv2.rectangle(bgr, (3, 3), (w - 4, h - 4), (0, 0, 255), int(border_thickness))
        cv2.putText(
            bgr,
            "FAILURE DETECTED",
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            float(banner_font_scale),
            (0, 0, 255),
            int(banner_thickness),
            cv2.LINE_AA,
        )

    line_1 = f"{detector_name} frame={int(frame_id) + 1}"
    line_2 = f"lambda={float(aggregate_score):.5f} threshold={float(threshold):.5f}"
    cv2.putText(
        bgr,
        line_1,
        (20, h - 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        float(footer_font_scale),
        (255, 255, 255),
        int(footer_thickness),
        cv2.LINE_AA,
    )
    cv2.putText(
        bgr,
        line_2,
        (20, h - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        float(footer_font_scale),
        (255, 255, 255),
        int(footer_thickness),
        cv2.LINE_AA,
    )

    if extra_text:
        cv2.putText(
            bgr,
            str(extra_text),
            (20, h - 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            float(footer_font_scale),
            (255, 255, 255),
            int(footer_thickness),
            cv2.LINE_AA,
        )

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
