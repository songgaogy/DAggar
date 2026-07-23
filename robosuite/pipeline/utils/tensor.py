from __future__ import annotations

import torch


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for all AWR tensor computation; CPU fallback is disabled.")


def resolve_cuda_device(requested: str | torch.device) -> torch.device:
    require_cuda()
    device = torch.device(requested)
    if device.type != "cuda":
        raise ValueError(f"AWR requires a CUDA device, got '{requested}'.")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError(
            f"CUDA device index {device.index} is unavailable; found {torch.cuda.device_count()} device(s)."
        )
    return device
