from __future__ import annotations

import os

import imageio


def create_vscode_mp4_writer(video_path: str, fps: int):
    """
    Create an mp4 writer that is easy for VSCode to preview.

    The preferred path is H.264 + yuv420p. If the local ffmpeg backend
    does not accept those parameters, fall back to imageio defaults.
    """
    os.makedirs(os.path.dirname(os.path.abspath(video_path)), exist_ok=True)
    try:
        return imageio.get_writer(
            video_path,
            fps=int(fps),
            format="FFMPEG",
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=1,
        )
    except Exception:
        return imageio.get_writer(video_path, fps=int(fps))
