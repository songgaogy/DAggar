import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cfg_get(cfg: Any, key: str, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half_dim = dim // 2
    freq = torch.exp(
        -math.log(10000.0)
        * torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
        / max(half_dim, 1)
    )
    args = timesteps.float().unsqueeze(-1) * freq.unsqueeze(0)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding


def _film(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)


def _group_norm_groups(num_channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if num_channels % groups == 0:
            return groups
    return 1


class FiLMResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, dropout: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = nn.GroupNorm(_group_norm_groups(in_channels), in_channels)
        self.norm2 = nn.GroupNorm(_group_norm_groups(out_channels), out_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.cond_proj = nn.Linear(cond_dim, 2 * in_channels + 2 * out_channels)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        film_1, film_2 = torch.split(
            self.cond_proj(cond),
            [2 * self.in_channels, 2 * self.out_channels],
            dim=-1,
        )
        scale_1, shift_1 = film_1.chunk(2, dim=-1)
        scale_2, shift_2 = film_2.chunk(2, dim=-1)

        h = _film(self.norm1(x), scale_1, shift_1)
        h = F.mish(h)
        h = self.conv1(h)
        h = _film(self.norm2(h), scale_2, shift_2)
        h = F.mish(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)


class Downsample1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] <= 1:
            return self.proj(x)
        return self.conv(x)


class Upsample1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, target_length: int) -> torch.Tensor:
        if x.shape[-1] != target_length:
            x = F.interpolate(x, size=target_length, mode="nearest")
        return self.conv(x)


class TimeConditioner(nn.Module):
    def __init__(self, time_dim: int, context_dim: int, cond_dim: int):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, cond_dim),
            nn.Mish(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim + context_dim, cond_dim),
            nn.Mish(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, timesteps: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        time_embedding = sinusoidal_time_embedding(timesteps, self.time_dim)
        time_features = self.time_mlp(time_embedding)
        return self.cond_mlp(torch.cat([time_features, context], dim=-1))


class UNet1DFlowHead(nn.Module):
    def __init__(
        self,
        action_dim: int,
        context_dim: int,
        hidden_dim: int,
        time_dim: int,
        cond_dim: int,
        channel_mults: list[int],
        dropout: float = 0.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.conditioner = TimeConditioner(time_dim=time_dim, context_dim=context_dim, cond_dim=cond_dim)

        channels = [hidden_dim * mult for mult in channel_mults]
        self.input_proj = nn.Conv1d(action_dim, channels[0], kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        current_channels = channels[0]
        for idx, stage_channels in enumerate(channels):
            blocks = nn.ModuleList(
                [
                    FiLMResidualBlock1D(current_channels, stage_channels, cond_dim, dropout=dropout),
                    FiLMResidualBlock1D(stage_channels, stage_channels, cond_dim, dropout=dropout),
                ]
            )
            self.down_blocks.append(blocks)
            current_channels = stage_channels
            if idx < len(channels) - 1:
                self.downsamples.append(Downsample1D(stage_channels, channels[idx + 1]))
                current_channels = channels[idx + 1]

        self.mid_blocks = nn.ModuleList(
            [
                FiLMResidualBlock1D(channels[-1], channels[-1], cond_dim, dropout=dropout),
                FiLMResidualBlock1D(channels[-1], channels[-1], cond_dim, dropout=dropout),
            ]
        )
        self.deepest_merge = nn.ModuleList(
            [
                FiLMResidualBlock1D(channels[-1] * 2, channels[-1], cond_dim, dropout=dropout),
                FiLMResidualBlock1D(channels[-1], channels[-1], cond_dim, dropout=dropout),
            ]
        )

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for idx in range(len(channels) - 2, -1, -1):
            in_channels = channels[idx + 1]
            out_channels = channels[idx]
            self.upsamples.append(Upsample1D(in_channels, out_channels))
            blocks = nn.ModuleList(
                [
                    FiLMResidualBlock1D(out_channels * 2, out_channels, cond_dim, dropout=dropout),
                    FiLMResidualBlock1D(out_channels, out_channels, cond_dim, dropout=dropout),
                ]
            )
            self.up_blocks.append(blocks)

        self.output_norm = nn.GroupNorm(_group_norm_groups(channels[0]), channels[0])
        self.output_conv = nn.Conv1d(channels[0], action_dim, kernel_size=3, padding=1)

    def forward(self, x_t: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        input_channels_first = True
        if x_t.ndim != 3:
            raise ValueError(f"Expected x_t to have shape [B, C, T] or [B, T, C], got {x_t.shape}")
        if x_t.shape[1] != self.action_dim and x_t.shape[2] == self.action_dim:
            x_t = x_t.transpose(1, 2)
            input_channels_first = False
        elif x_t.shape[1] != self.action_dim:
            raise ValueError(f"Input action dim mismatch: expected {self.action_dim}, got shape {x_t.shape}")

        cond = self.conditioner(timesteps, context)
        h = self.input_proj(x_t)

        skips = []
        for idx, blocks in enumerate(self.down_blocks):
            for block in blocks:
                h = block(h, cond)
            skips.append(h)
            if idx < len(self.downsamples):
                h = self.downsamples[idx](h)

        for block in self.mid_blocks:
            h = block(h, cond)

        deepest_skip = skips.pop()
        h = torch.cat([h, deepest_skip], dim=1)
        for block in self.deepest_merge:
            h = block(h, cond)

        for upsample, blocks in zip(self.upsamples, self.up_blocks):
            skip = skips.pop()
            h = upsample(h, target_length=skip.shape[-1])
            h = torch.cat([h, skip], dim=1)
            for block in blocks:
                h = block(h, cond)

        h = F.mish(self.output_norm(h))
        out = self.output_conv(h)
        if not input_channels_first:
            out = out.transpose(1, 2)
        return out


def build_flow_head(cfg: Any, action_dim: int, context_dim: int) -> nn.Module:
    head_type = _cfg_get(cfg, "type", "unet_1d")
    if head_type != "unet_1d":
        raise ValueError(f"Unsupported flow head type: {head_type}")
    channel_mults = [int(mult) for mult in _cfg_get(cfg, "channel_mults", [1, 2, 2, 2])]
    return UNet1DFlowHead(
        action_dim=action_dim,
        context_dim=context_dim,
        hidden_dim=int(_cfg_get(cfg, "hidden_dim", 256)),
        time_dim=int(_cfg_get(cfg, "time_dim", 256)),
        cond_dim=int(_cfg_get(cfg, "cond_dim", 512)),
        channel_mults=channel_mults,
        dropout=float(_cfg_get(cfg, "dropout", 0.0)),
    )
