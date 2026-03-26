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


def _draw_text_with_box(
    image_bgr: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    font_scale: float,
    thickness: int,
    text_color: tuple[int, int, int],
    box_color: tuple[int, int, int],
    outline_color: Optional[tuple[int, int, int]] = None,
    padding_x: int = 8,
    padding_y: int = 6,
) -> None:
    if cv2 is None:
        return
    x, y = int(origin[0]), int(origin[1])
    fs = float(font_scale)
    th = max(1, int(thickness))
    (text_w, text_h), baseline = cv2.getTextSize(
        str(text),
        cv2.FONT_HERSHEY_SIMPLEX,
        fs,
        th,
    )
    x0 = max(0, x - int(padding_x))
    y0 = max(0, y - text_h - int(padding_y))
    x1 = min(int(image_bgr.shape[1]) - 1, x + text_w + int(padding_x))
    y1 = min(int(image_bgr.shape[0]) - 1, y + baseline + int(padding_y))
    cv2.rectangle(image_bgr, (x0, y0), (x1, y1), box_color, -1)
    if outline_color is not None:
        outline_thickness = max(2, th + 2)
        cv2.putText(
            image_bgr,
            str(text),
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            fs,
            outline_color,
            outline_thickness,
            cv2.LINE_AA,
        )
    cv2.putText(
        image_bgr,
        str(text),
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        fs,
        text_color,
        th,
        cv2.LINE_AA,
    )


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
    footer_lines: Optional[Sequence[str]] = None,
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
        _draw_text_with_box(
            bgr,
            "FAILURE DETECTED",
            (18, 30),
            font_scale=float(banner_font_scale),
            thickness=int(banner_thickness),
            text_color=(255, 255, 255),
            box_color=(0, 0, 180),
            outline_color=(0, 0, 0),
        )

    if footer_lines is not None:
        lines = [str(line) for line in footer_lines if str(line)]
    else:
        lines = [
            f"{detector_name} frame={int(frame_id) + 1}",
            f"lambda={float(aggregate_score):.5f} threshold={float(threshold):.5f}",
        ]
        if extra_text:
            lines.insert(0, str(extra_text))

    if lines:
        line_gap = max(20, int(28 * float(footer_font_scale)))
        start_y = h - 16 - (len(lines) - 1) * line_gap
        for idx, line in enumerate(lines):
            _draw_text_with_box(
                bgr,
                line,
                (20, start_y + idx * line_gap),
                font_scale=float(footer_font_scale),
                thickness=int(footer_thickness),
                text_color=(255, 255, 255),
                box_color=(0, 0, 0),
                outline_color=(32, 32, 32),
            )

    elif extra_text:
        _draw_text_with_box(
            bgr,
            str(extra_text),
            (20, h - 16),
            font_scale=float(footer_font_scale),
            thickness=int(footer_thickness),
            text_color=(255, 255, 255),
            box_color=(0, 0, 0),
            outline_color=(32, 32, 32),
        )

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
