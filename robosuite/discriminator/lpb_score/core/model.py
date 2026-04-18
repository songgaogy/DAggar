"""Single-head chunk DSM with LoRA encoder tuning and temporal Conv1d aggregation."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn

from robosuite.discriminator.dyn_bce.modules.flow_encoder import FlowMultitaskEncoder

from .dataset import PreparedTrajectory


MODEL_ARCHITECTURE = "joint_lora_encoder_single_head_temporal_conv_chunk_state_only_conditional_dsm"


def _cfg_get(cfg: Any, key: str, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class AdaLNModulation(nn.Module):
    def __init__(self, cond_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(cond_dim, 2 * embed_dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shift, scale = self.linear(cond).chunk(2, dim=-1)
        return shift, scale


class AdaLNTemporalConvBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        cond_dim: int,
        ffn_dim: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if int(kernel_size) <= 0 or int(kernel_size) % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.mod = AdaLNModulation(cond_dim=cond_dim, embed_dim=embed_dim)
        padding = int(kernel_size) // 2
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, ffn_dim, kernel_size=int(kernel_size), padding=padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(ffn_dim, embed_dim, kernel_size=int(kernel_size), padding=padding),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.mod(cond)
        y = self.norm(x.transpose(1, 2)).transpose(1, 2)
        y = y * (1.0 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        return x + self.conv(y)


class ChunkConditionedDSM(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        window_size: int,
        num_tasks: int,
        embed_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.window_size = int(window_size)
        self.num_tasks = int(num_tasks)
        self.embed_dim = int(embed_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.ffn_dim = int(ffn_dim)
        self.kernel_size = int(kernel_size)
        self.dropout = float(dropout)
        self.chunk_dim = int(self.latent_dim * self.window_size)

        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive.")
        if self.num_tasks <= 0:
            raise ValueError("num_tasks must be positive.")

        self.type_embedding = nn.Embedding(2, self.embed_dim)
        self.task_embedding = nn.Embedding(self.num_tasks, self.embed_dim)
        self.condition_mlp = self._build_condition_mlp(self.embed_dim, self.embed_dim)
        self.input_proj = self._build_input_proj(self.latent_dim)
        self.blocks = nn.ModuleList(
            [
                AdaLNTemporalConvBlock(
                    embed_dim=self.embed_dim,
                    cond_dim=self.embed_dim,
                    ffn_dim=self.ffn_dim,
                    kernel_size=self.kernel_size,
                    dropout=self.dropout,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(self.embed_dim)
        self.chunk_head = nn.Linear(self.embed_dim, self.latent_dim)

    def _build_input_proj(self, input_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

    def _build_condition_mlp(self, input_dim: int, output_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, self.embed_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embed_dim, output_dim),
        )

    def _require_latent_window(self, name: str, x: torch.Tensor) -> None:
        if x.ndim != 3:
            raise ValueError(f"Expected {name} shape (B,W,D), got {tuple(x.shape)}")
        if int(x.shape[1]) != self.window_size:
            raise ValueError(f"{name} window mismatch: expected {self.window_size}, got {x.shape[1]}")
        if int(x.shape[2]) != self.latent_dim:
            raise ValueError(f"{name} latent dim mismatch: expected {self.latent_dim}, got {x.shape[2]}")

    def _require_task_index(
        self,
        batch_size: int,
        task_index: torch.Tensor | int,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(task_index, int):
            task_index_tensor = torch.full((batch_size,), int(task_index), dtype=torch.long, device=device)
        else:
            task_index_tensor = torch.as_tensor(task_index, dtype=torch.long, device=device).reshape(-1)
        if int(task_index_tensor.shape[0]) != int(batch_size):
            raise ValueError(f"Expected task_index shape ({batch_size},), got {tuple(task_index_tensor.shape)}")
        if torch.any((task_index_tensor < 0) | (task_index_tensor >= self.num_tasks)):
            raise ValueError(f"task_index must be in [0, {self.num_tasks - 1}]")
        return task_index_tensor

    @staticmethod
    def _require_traj_type(
        batch_size: int,
        traj_type: torch.Tensor | int,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(traj_type, int):
            traj_type_tensor = torch.full((batch_size,), int(traj_type), dtype=torch.long, device=device)
        else:
            traj_type_tensor = torch.as_tensor(traj_type, dtype=torch.long, device=device).reshape(-1)
        if int(traj_type_tensor.shape[0]) != int(batch_size):
            raise ValueError(f"Expected traj_type shape ({batch_size},), got {tuple(traj_type_tensor.shape)}")
        if torch.any((traj_type_tensor < 0) | (traj_type_tensor > 1)):
            raise ValueError("traj_type must contain only 0 or 1.")
        return traj_type_tensor

    def _build_condition(
        self,
        *,
        traj_type: torch.Tensor,
        task_index: torch.Tensor,
    ) -> torch.Tensor:
        cond_embed = self.type_embedding(traj_type) + self.task_embedding(task_index)
        return self.condition_mlp(cond_embed)

    @staticmethod
    def _module_parameters(*modules: nn.Module) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for module in modules:
            params.extend(list(module.parameters()))
        return params

    def optimizer_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        return {
            "shared": self._module_parameters(
                self.type_embedding,
                self.task_embedding,
                self.condition_mlp,
                self.input_proj,
                self.blocks,
                self.final_norm,
            ),
            "chunk_branch": self._module_parameters(self.chunk_head),
        }

    def forward(
        self,
        *,
        latent_window_input: torch.Tensor,
        traj_type: torch.Tensor,
        task_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size = int(latent_window_input.shape[0])
        self._require_latent_window("latent_window_input", latent_window_input)

        traj_type_tensor = self._require_traj_type(
            batch_size=batch_size,
            traj_type=traj_type,
            device=latent_window_input.device,
        )
        task_index_tensor = self._require_task_index(
            batch_size=batch_size,
            task_index=task_index,
            device=latent_window_input.device,
        )

        cond = self._build_condition(
            traj_type=traj_type_tensor,
            task_index=task_index_tensor,
        )
        hidden = self.input_proj(latent_window_input).transpose(1, 2)
        for block in self.blocks:
            hidden = block(hidden, cond)
        hidden = self.final_norm(hidden.transpose(1, 2))
        return {"latent_window_hat": self.chunk_head(hidden)}


UnifiedConditionedDSM = ChunkConditionedDSM
ConditionalManifoldDenoiser = ChunkConditionedDSM
JointManifoldDenoiser = ChunkConditionedDSM


class DSMModel(nn.Module):
    def __init__(
        self,
        predictor: ChunkConditionedDSM,
        policy_encoder: FlowMultitaskEncoder,
        task_names: Sequence[str],
        latent_dim: int,
        window_size: int,
        noise_scale: float,
        std_clamp_min: float,
    ) -> None:
        super().__init__()
        self.predictor = predictor
        self.policy_encoder = policy_encoder
        self.task_names = [str(name) for name in task_names]
        self.latent_dim = int(latent_dim)
        self.window_size = int(window_size)
        self.noise_scale = float(noise_scale)
        self.noise_sigma = self.noise_scale
        self.std_clamp_min = float(std_clamp_min)
        self.num_tasks = int(self.predictor.num_tasks)
        self.chunk_dim = int(self.latent_dim * self.window_size)

        self.register_buffer("latent_mean", torch.zeros(self.latent_dim, dtype=torch.float32))
        self.register_buffer("latent_var", torch.ones(self.latent_dim, dtype=torch.float32))

        if self.num_tasks != len(self.task_names):
            raise ValueError(f"task_names length mismatch: expected {self.num_tasks}, got {len(self.task_names)}")
        if int(self.predictor.latent_dim) != self.latent_dim:
            raise ValueError(
                f"predictor latent_dim mismatch: expected {self.latent_dim}, got {self.predictor.latent_dim}"
            )
        if int(self.predictor.window_size) != self.window_size:
            raise ValueError(
                f"predictor window_size mismatch: expected {self.window_size}, got {self.predictor.window_size}"
            )
        if int(self.policy_encoder.latent_dim) != self.latent_dim:
            raise ValueError(
                f"policy encoder latent_dim mismatch: expected {self.latent_dim}, got {self.policy_encoder.latent_dim}"
            )

    def optimizer_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        groups = self.predictor.optimizer_parameter_groups()
        groups["encoder_branch"] = list(self.policy_encoder.trainable_parameters())
        return groups

    def set_normalization_stats(
        self,
        *,
        latent_mean: torch.Tensor,
        latent_var: torch.Tensor,
        min_variance: float = 1e-6,
    ) -> None:
        latent_mean = torch.as_tensor(latent_mean, dtype=torch.float32, device=self.latent_mean.device).reshape(-1)
        latent_var = torch.as_tensor(latent_var, dtype=torch.float32, device=self.latent_var.device).reshape(-1)
        if latent_mean.shape[0] != self.latent_dim or latent_var.shape[0] != self.latent_dim:
            raise ValueError(
                f"Latent stats must have shape ({self.latent_dim},), "
                f"got mean={tuple(latent_mean.shape)} var={tuple(latent_var.shape)}"
            )
        self.latent_mean.copy_(latent_mean)
        self.latent_var.copy_(torch.clamp(latent_var, min=float(min_variance)))

    @property
    def latent_std(self) -> torch.Tensor:
        return torch.clamp(torch.sqrt(self.latent_var), min=float(self.std_clamp_min))

    def normalize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        latent_mean = self.latent_mean.to(device=latent.device, dtype=latent.dtype)
        latent_std = self.latent_std.to(device=latent.device, dtype=latent.dtype)
        shape = [1] * latent.ndim
        shape[-1] = self.latent_dim
        return (latent - latent_mean.view(*shape)) / latent_std.view(*shape)

    def flatten_latent_window(self, latent_window: torch.Tensor) -> torch.Tensor:
        if latent_window.ndim != 3:
            raise ValueError(f"Expected latent_window shape (B,W,D), got {tuple(latent_window.shape)}")
        if int(latent_window.shape[1]) != self.window_size:
            raise ValueError(
                f"latent_window window mismatch: expected {self.window_size}, got {latent_window.shape[1]}"
            )
        if int(latent_window.shape[2]) != self.latent_dim:
            raise ValueError(
                f"latent_window dim mismatch: expected {self.latent_dim}, got {latent_window.shape[2]}"
            )
        return latent_window.reshape(latent_window.shape[0], self.chunk_dim)

    def add_noise(self, *, chunk_clean: torch.Tensor) -> dict[str, torch.Tensor]:
        chunk_noise = torch.randn_like(chunk_clean) * float(self.noise_scale)
        return {
            "chunk_input": chunk_clean + chunk_noise,
            "chunk_noise": chunk_noise,
        }

    @staticmethod
    def _require_traj_type(batch_size: int, traj_type: torch.Tensor | int, device: torch.device) -> torch.Tensor:
        if isinstance(traj_type, int):
            return torch.full((batch_size,), int(traj_type), dtype=torch.long, device=device)
        traj_type_tensor = torch.as_tensor(traj_type, dtype=torch.long, device=device).reshape(-1)
        if int(traj_type_tensor.shape[0]) != int(batch_size):
            raise ValueError(f"Expected traj_type length {batch_size}, got {traj_type_tensor.shape[0]}")
        if torch.any((traj_type_tensor < 0) | (traj_type_tensor > 1)):
            raise ValueError("traj_type must contain only 0 or 1.")
        return traj_type_tensor

    def _require_task_index(
        self,
        batch_size: int,
        task_index: torch.Tensor | int,
        device: torch.device,
    ) -> torch.Tensor:
        return self.predictor._require_task_index(
            batch_size=batch_size,
            task_index=task_index,
            device=device,
        )

    def _task_names_from_index(
        self,
        *,
        task_index: torch.Tensor,
        repeat_each: int = 1,
    ) -> list[str]:
        names: list[str] = []
        for idx in task_index.detach().cpu().tolist():
            names.extend([self.task_names[int(idx)]] * int(repeat_each))
        return names

    def encode_latent_window(
        self,
        *,
        image_window: torch.Tensor,
        proprio_window: torch.Tensor,
        task_index: torch.Tensor | int,
    ) -> torch.Tensor:
        if image_window.ndim != 6:
            raise ValueError(f"Expected image_window shape (B,W,V,3,H,W), got {tuple(image_window.shape)}")
        if proprio_window.ndim != 3:
            raise ValueError(f"Expected proprio_window shape (B,W,P), got {tuple(proprio_window.shape)}")
        if int(image_window.shape[0]) != int(proprio_window.shape[0]) or int(image_window.shape[1]) != int(
            proprio_window.shape[1]
        ):
            raise ValueError("image_window and proprio_window batch/window dims must match.")
        if int(image_window.shape[1]) != self.window_size:
            raise ValueError(
                f"image_window window mismatch: expected {self.window_size}, got {image_window.shape[1]}"
            )

        batch_size = int(image_window.shape[0])
        task_index_tensor = self._require_task_index(
            batch_size=batch_size,
            task_index=task_index,
            device=image_window.device,
        )
        images_flat = image_window.reshape(
            batch_size * self.window_size,
            int(image_window.shape[2]),
            int(image_window.shape[3]),
            int(image_window.shape[4]),
            int(image_window.shape[5]),
        )
        proprio_flat = proprio_window.reshape(batch_size * self.window_size, int(proprio_window.shape[2]))
        task_names = self._task_names_from_index(task_index=task_index_tensor, repeat_each=self.window_size)
        latents_flat = self.policy_encoder.encode_context_tensors(
            images=images_flat,
            proprio=proprio_flat,
            task_names=task_names,
        )
        return latents_flat.reshape(batch_size, self.window_size, self.latent_dim)

    def encode_trajectory(
        self,
        *,
        images: torch.Tensor | Sequence,
        proprio: torch.Tensor | Sequence,
        task_index: torch.Tensor | int,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        image_tensor = torch.as_tensor(images, dtype=torch.float32, device=self.latent_mean.device)
        proprio_tensor = torch.as_tensor(proprio, dtype=torch.float32, device=self.latent_mean.device)
        if image_tensor.ndim != 5 or proprio_tensor.ndim != 2:
            raise ValueError("encode_trajectory expects images (T,V,3,H,W) and proprio (T,P).")
        if int(image_tensor.shape[0]) != int(proprio_tensor.shape[0]):
            raise ValueError("encode_trajectory images and proprio lengths must match.")

        total_steps = int(image_tensor.shape[0])
        task_index_tensor = self._require_task_index(
            batch_size=1,
            task_index=int(task_index) if isinstance(task_index, int) else torch.as_tensor(task_index).reshape(-1)[0],
            device=image_tensor.device,
        )
        task_name = self.task_names[int(task_index_tensor[0].item())]
        batch = max(1, int(batch_size or self.policy_encoder.batch_size))
        outputs: list[torch.Tensor] = []
        for start in range(0, total_steps, batch):
            end = min(start + batch, total_steps)
            outputs.append(
                self.policy_encoder.encode_context_tensors(
                    images=image_tensor[start:end],
                    proprio=proprio_tensor[start:end],
                    task_names=[task_name] * int(end - start),
                )
            )
        return torch.cat(outputs, dim=0)

    @torch.no_grad()
    def compute_normalization_stats(
        self,
        trajectories: Iterable[PreparedTrajectory],
        *,
        batch_size: int | None = None,
        min_variance: float = 1e-6,
    ) -> dict[str, torch.Tensor]:
        was_training = self.training
        self.eval()

        latent_sum = torch.zeros((self.latent_dim,), dtype=torch.float64, device=self.latent_mean.device)
        latent_sq_sum = torch.zeros((self.latent_dim,), dtype=torch.float64, device=self.latent_mean.device)
        latent_count = 0

        for traj in trajectories:
            latents = self.encode_trajectory(
                images=traj.images,
                proprio=traj.proprio,
                task_index=int(traj.task_index),
                batch_size=batch_size,
            ).to(dtype=torch.float64)
            latent_sum += latents.sum(dim=0)
            latent_sq_sum += torch.square(latents).sum(dim=0)
            latent_count += int(latents.shape[0])

        if was_training:
            self.train()
        if latent_count <= 0:
            raise RuntimeError("Positive normalization stats require positive latent counts.")

        latent_mean = latent_sum / float(latent_count)
        latent_var = torch.clamp(
            latent_sq_sum / float(latent_count) - torch.square(latent_mean),
            min=float(min_variance),
        )
        return {
            "latent_mean": latent_mean.to(dtype=torch.float32).detach().cpu(),
            "latent_var": latent_var.to(dtype=torch.float32).detach().cpu(),
            "latent_count": torch.tensor(int(latent_count), dtype=torch.int64),
        }

    def reconstruction_components(
        self,
        *,
        chunk_clean: torch.Tensor,
        chunk_hat: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        chunk_sq = torch.square(chunk_hat - chunk_clean)
        chunk_energy = chunk_sq.mean(dim=-1)
        return {
            "error_sq": chunk_sq,
            "chunk_mse_per_sample": chunk_energy,
            "chunk_sse_per_sample": chunk_sq.sum(dim=-1),
            "chunk_energy_per_sample": chunk_energy,
            "score_per_sample": chunk_energy,
        }

    def forward_from_latent_window(
        self,
        latent_window: torch.Tensor,
        *,
        traj_type: torch.Tensor | int,
        task_index: torch.Tensor | int,
        add_noise: bool = False,
    ) -> dict[str, torch.Tensor]:
        latent_window_clean = self.normalize_latent(latent_window)
        latent_window_input = latent_window_clean
        latent_window_noise = torch.zeros_like(latent_window_clean)
        if add_noise:
            latent_window_noise = torch.randn_like(latent_window_clean) * float(self.noise_scale)
            latent_window_input = latent_window_clean + latent_window_noise
        chunk_clean = self.flatten_latent_window(latent_window_clean)
        chunk_input = self.flatten_latent_window(latent_window_input)
        chunk_noise = self.flatten_latent_window(latent_window_noise)

        traj_type_tensor = self._require_traj_type(
            batch_size=int(latent_window.shape[0]),
            traj_type=traj_type,
            device=latent_window.device,
        )
        task_index_tensor = self._require_task_index(
            batch_size=int(latent_window.shape[0]),
            task_index=task_index,
            device=latent_window.device,
        )

        preds = self.predictor(
            latent_window_input=latent_window_input,
            traj_type=traj_type_tensor,
            task_index=task_index_tensor,
        )
        latent_window_hat = preds["latent_window_hat"]
        return {
            "latent_window": latent_window,
            "latent_window_clean": latent_window_clean,
            "latent_window_input": latent_window_input,
            "latent_window_hat": latent_window_hat,
            "traj_type": traj_type_tensor,
            "task_index": task_index_tensor,
            "chunk_clean": chunk_clean,
            "chunk_input": chunk_input,
            "chunk_hat": self.flatten_latent_window(latent_window_hat),
            "chunk_noise": chunk_noise,
        }

    def forward(
        self,
        image_window: torch.Tensor,
        proprio_window: torch.Tensor,
        *,
        traj_type: torch.Tensor | int,
        task_index: torch.Tensor | int,
        add_noise: bool = False,
    ) -> dict[str, torch.Tensor]:
        latent_window = self.encode_latent_window(
            image_window=image_window,
            proprio_window=proprio_window,
            task_index=task_index,
        )
        out = self.forward_from_latent_window(
            latent_window=latent_window,
            traj_type=traj_type,
            task_index=task_index,
            add_noise=add_noise,
        )
        out["image_window"] = image_window
        out["proprio_window"] = proprio_window
        return out

    def compute_dsm_loss(
        self,
        image_window: torch.Tensor,
        proprio_window: torch.Tensor,
        traj_type: torch.Tensor,
        task_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        out = self.forward(
            image_window=image_window,
            proprio_window=proprio_window,
            traj_type=traj_type,
            task_index=task_index,
            add_noise=True,
        )
        recon = self.reconstruction_components(
            chunk_clean=out["chunk_clean"],
            chunk_hat=out["chunk_hat"],
        )
        loss = recon["chunk_energy_per_sample"].mean()
        return {
            "loss": loss,
            "score": loss,
            "unweighted_score": loss,
            "chunk_mse": recon["chunk_mse_per_sample"].mean(),
            "chunk_energy": recon["chunk_energy_per_sample"].mean(),
            "latent_window": out["latent_window"],
            "traj_type": out["traj_type"],
            "task_index": out["task_index"],
            "chunk_clean": out["chunk_clean"],
            "chunk_input": out["chunk_input"],
            "chunk_hat": out["chunk_hat"],
        }

    def compute_fisher_score_from_latent_window(
        self,
        *,
        latent_window: torch.Tensor,
        task_index: torch.Tensor | int,
    ) -> dict[str, torch.Tensor]:
        pos_out = self.forward_from_latent_window(
            latent_window=latent_window,
            traj_type=0,
            task_index=task_index,
            add_noise=False,
        )
        neg_out = self.forward_from_latent_window(
            latent_window=latent_window,
            traj_type=1,
            task_index=task_index,
            add_noise=False,
        )
        pos_recon = self.reconstruction_components(
            chunk_clean=pos_out["chunk_clean"],
            chunk_hat=pos_out["chunk_hat"],
        )
        neg_recon = self.reconstruction_components(
            chunk_clean=pos_out["chunk_clean"],
            chunk_hat=neg_out["chunk_hat"],
        )
        chunk_margin = neg_recon["chunk_energy_per_sample"] - pos_recon["chunk_energy_per_sample"]
        return {
            "chunk_positive_energy_per_sample": pos_recon["chunk_energy_per_sample"],
            "chunk_negative_energy_per_sample": neg_recon["chunk_energy_per_sample"],
            "chunk_margin_per_sample": chunk_margin,
            "margin_score_per_sample": chunk_margin,
            "positive_score_per_sample": pos_recon["score_per_sample"],
            "negative_score_per_sample": neg_recon["score_per_sample"],
            "chunk_pos_hat": pos_out["chunk_hat"],
            "chunk_neg_hat": neg_out["chunk_hat"],
        }

    def compute_fisher_score(
        self,
        *,
        image_window: torch.Tensor,
        proprio_window: torch.Tensor,
        task_index: torch.Tensor | int,
    ) -> dict[str, torch.Tensor]:
        latent_window = self.encode_latent_window(
            image_window=image_window,
            proprio_window=proprio_window,
            task_index=task_index,
        )
        return self.compute_fisher_score_from_latent_window(
            latent_window=latent_window,
            task_index=task_index,
        )


def build_unified_conditioned_dsm(
    *,
    latent_dim: int,
    window_size: int,
    num_tasks: int,
    cfg_model: Any,
) -> ChunkConditionedDSM:
    embed_dim = int(_cfg_get(cfg_model, "embed_dim", _cfg_get(cfg_model, "backbone_dim", 512)))
    return ChunkConditionedDSM(
        latent_dim=int(latent_dim),
        window_size=int(window_size),
        num_tasks=int(num_tasks),
        embed_dim=embed_dim,
        num_layers=int(_cfg_get(cfg_model, "num_layers", _cfg_get(cfg_model, "backbone_num_blocks", 4))),
        num_heads=int(_cfg_get(cfg_model, "num_heads", 8)),
        ffn_dim=int(_cfg_get(cfg_model, "ffn_dim", max(4 * embed_dim, embed_dim))),
        kernel_size=int(_cfg_get(cfg_model, "kernel_size", 3)),
        dropout=float(_cfg_get(cfg_model, "dropout", 0.1)),
    )


def build_conditional_manifold_denoiser(
    *,
    latent_dim: int,
    num_tasks: int,
    cfg_model: Any,
    window_size: int | None = None,
    action_flat_dim: int | None = None,
) -> ChunkConditionedDSM:
    if window_size is None:
        if action_flat_dim is None or int(action_flat_dim) <= 0 or int(action_flat_dim) % int(latent_dim) != 0:
            raise ValueError("Provide window_size for the chunk DSM predictor.")
        window_size = int(action_flat_dim) // int(latent_dim)
    return build_unified_conditioned_dsm(
        latent_dim=int(latent_dim),
        window_size=int(window_size),
        num_tasks=int(num_tasks),
        cfg_model=cfg_model,
    )


def build_joint_manifold_denoiser(
    *,
    tau_dim: int,
    cfg_model: Any,
) -> ChunkConditionedDSM:
    raise ValueError(
        "build_joint_manifold_denoiser requires the removed joint interface. "
        "Use build_unified_conditioned_dsm with latent_dim and window_size."
    )


def build_dsm_model(
    *,
    latent_dim: int,
    num_tasks: int,
    cfg_model: Any,
    window_size: int,
    policy_encoder: FlowMultitaskEncoder,
    task_names: Sequence[str],
) -> DSMModel:
    predictor = build_unified_conditioned_dsm(
        latent_dim=int(latent_dim),
        window_size=int(window_size),
        num_tasks=int(num_tasks),
        cfg_model=cfg_model,
    )
    return DSMModel(
        predictor=predictor,
        policy_encoder=policy_encoder,
        task_names=task_names,
        latent_dim=int(latent_dim),
        window_size=int(window_size),
        noise_scale=float(_cfg_get(cfg_model, "noise_scale", _cfg_get(cfg_model, "noise_sigma", 0.08))),
        std_clamp_min=float(_cfg_get(cfg_model, "std_clamp_min", 0.05)),
    )
