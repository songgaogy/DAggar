from __future__ import annotations

from typing import Mapping, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class RandomShiftsAug(nn.Module):
    """Integer random-shift augmentation with replicate padding."""

    def __init__(self, pad: int = 4) -> None:
        super().__init__()
        self.pad = int(pad)

    def sample_shifts(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(
            0,
            2 * self.pad + 1,
            (int(batch_size), 2),
            device=device,
        )

    def forward(self, images: torch.Tensor, shifts: torch.Tensor | None = None) -> torch.Tensor:
        if images.dim() != 4:
            raise ValueError(f"Expected images shaped (N, C, H, W), got {tuple(images.shape)}")
        n, c, height, width = images.shape
        if shifts is None:
            shifts = self.sample_shifts(n, images.device)
        if shifts.shape != (n, 2):
            raise ValueError(f"Expected shifts shaped {(n, 2)}, got {tuple(shifts.shape)}")
        if self.pad == 0:
            return images

        padded = F.pad(images, (self.pad,) * 4, mode="replicate")
        padded_width = width + 2 * self.pad
        rows = torch.arange(height, device=images.device).view(1, height, 1)
        cols = torch.arange(width, device=images.device).view(1, 1, width)
        rows = rows + shifts[:, 0].view(n, 1, 1)
        cols = cols + shifts[:, 1].view(n, 1, 1)
        flat_indices = (rows * padded_width + cols).expand(n, height, width).reshape(n, 1, -1)
        flat_indices = flat_indices.expand(n, c, height * width)
        return padded.flatten(2).gather(2, flat_indices).reshape(n, c, height, width)


class TACOActionEncoder(nn.Module):
    """Encode a fixed action sequence using the TACO paper architecture."""

    def __init__(
        self,
        action_dim: int = 7,
        sequence_length: int = 8,
        step_hidden: int = 64,
        step_emb_dim: int = 9,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.sequence_length = int(sequence_length)
        self.step_hidden = int(step_hidden)
        self.step_emb_dim = int(step_emb_dim)
        self.in_chans = self.action_dim * self.sequence_length
        self.emb_dim = self.step_emb_dim * self.sequence_length
        self.step_encoder = nn.Sequential(
            nn.Linear(self.action_dim, self.step_hidden),
            nn.Tanh(),
            nn.Linear(self.step_hidden, self.step_emb_dim),
        )
        self.sequence_encoder = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.LayerNorm(self.emb_dim),
            nn.Tanh(),
        )

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.shape[-1] == self.in_chans:
            sequence = actions.unflatten(-1, (self.sequence_length, self.action_dim))
        elif actions.dim() >= 2 and tuple(actions.shape[-2:]) == (
            self.sequence_length,
            self.action_dim,
        ):
            sequence = actions
        else:
            raise ValueError(
                "Expected flattened action windows ending in "
                f"{self.in_chans}, or sequences ending in "
                f"({self.sequence_length}, {self.action_dim}); got {tuple(actions.shape)}"
            )
        step_latents = self.step_encoder(sequence)
        return self.sequence_encoder(step_latents.flatten(-2))


class TACORepresentationModel(nn.Module):
    """Temporal action-driven contrastive representation model.

    The visual encoder must expose ``emb_dim``, ``num_patches`` and
    ``_encode_flat(images)``. This matches :class:`DINOv3Encoder` while allowing
    a small injected encoder in CUDA unit tests.
    """

    def __init__(
        self,
        image_size: int,
        num_hist: int,
        num_pred: int,
        encoder: nn.Module,
        proprio_encoder: nn.Module,
        proprio_dim: int,
        action_dim_per_step: int = 7,
        frameskip: int = 8,
        view_names: Sequence[str] = ("agentview", "robot0_eye_in_hand"),
        source_view_names: Sequence[str] | None = None,
        target_view_names: Sequence[str] | None = None,
        state_dim: int = 50,
        transition_hidden: int = 1024,
        action_step_hidden: int = 64,
        action_step_emb_dim: int = 9,
        encoder_micro_batch_size: int = 128,
        random_shift_pad: int = 4,
        use_layernorm: bool = True,
        train_encoder: bool = False,
    ) -> None:
        super().__init__()
        if int(num_hist) != 1 or int(num_pred) != 1:
            raise ValueError("TACORepresentationModel currently requires num_hist=num_pred=1")
        if int(frameskip) <= 0:
            raise ValueError("frameskip must be positive")
        self.image_size = int(image_size)
        self.num_hist = int(num_hist)
        self.num_pred = int(num_pred)
        self.encoder = encoder
        self.proprio_encoder = proprio_encoder
        self.proprio_dim = int(proprio_dim)
        self.frameskip = int(frameskip)
        self.state_dim = int(state_dim)
        self.encoder_micro_batch_size = int(encoder_micro_batch_size)
        self.view_names = list(view_names)
        self.source_view_names = list(source_view_names or self.view_names)
        self.target_view_names = list(target_view_names or self.source_view_names)
        if not set(self.target_view_names).issubset(self.source_view_names):
            raise ValueError("target_view_names must be a subset of source_view_names")
        if not hasattr(encoder, "_encode_flat"):
            raise TypeError("encoder must provide _encode_flat(images)")
        if not hasattr(encoder, "emb_dim") or not hasattr(encoder, "num_patches"):
            raise TypeError("encoder must expose emb_dim and num_patches")

        self.action_encoder = TACOActionEncoder(
            action_dim_per_step,
            self.frameskip,
            step_hidden=action_step_hidden,
            step_emb_dim=action_step_emb_dim,
        )
        self.random_shift = RandomShiftsAug(random_shift_pad)
        self.per_view_norm = nn.ModuleDict(
            {
                name: nn.LayerNorm(int(encoder.emb_dim), elementwise_affine=False)
                for name in self.view_names
            }
        ) if use_layernorm else nn.ModuleDict()

        self._visual_block_dim = int(encoder.num_patches) * int(encoder.emb_dim)
        state_input_dim = self._visual_block_dim * len(self.source_view_names) + self.proprio_dim
        self.state_projector = nn.Linear(state_input_dim, self.state_dim)
        self.state_norm = nn.Sequential(
            nn.LayerNorm(self.state_dim),
            nn.Tanh(),
        )
        # Query and future share this projector. Selecting only target-view
        # columns is exactly equivalent to zero-filling unavailable future views.
        future_columns = []
        for name in self.target_view_names:
            view_idx = self.source_view_names.index(name)
            start = view_idx * self._visual_block_dim
            future_columns.extend(range(start, start + self._visual_block_dim))
        proprio_start = self._visual_block_dim * len(self.source_view_names)
        future_columns.extend(range(proprio_start, proprio_start + self.proprio_dim))
        self.register_buffer(
            "future_projector_columns",
            torch.tensor(future_columns, dtype=torch.long),
            persistent=False,
        )
        self.transition_hidden = int(transition_hidden)
        self.transition = nn.Sequential(
            nn.Linear(self.state_dim + self.action_encoder.emb_dim, self.transition_hidden),
            nn.ReLU(),
            nn.Linear(self.transition_hidden, self.state_dim),
        )
        self.W = nn.Parameter(torch.empty(self.state_dim, self.state_dim))
        nn.init.orthogonal_(self.W)

        if hasattr(self.encoder, "set_trainable"):
            train_projection = bool(getattr(self.encoder, "train_projection", True))
            self.encoder.set_trainable(train_backbone=bool(train_encoder), train_projection=train_projection)

    def train(self, mode: bool = True):
        super().train(mode)
        if bool(getattr(self.encoder, "freeze_backbone", False)) and hasattr(self.encoder, "backbone"):
            self.encoder.backbone.eval()
        return self

    @staticmethod
    def _float_images(images: torch.Tensor) -> torch.Tensor:
        if images.dtype == torch.uint8:
            return images.to(torch.float32).div_(255.0)
        return images

    def _augment_views(
        self,
        visual: Mapping[str, torch.Tensor],
        view_names: Sequence[str],
        time_index: int,
    ) -> dict[str, torch.Tensor]:
        first = visual[view_names[0]][:, time_index]
        shifts = self.random_shift.sample_shifts(first.shape[0], first.device)
        return {
            name: self.random_shift(self._float_images(visual[name][:, time_index]), shifts)
            for name in view_names
        }

    def _encode_flat_microbatched(self, images: torch.Tensor) -> torch.Tensor:
        chunk_size = self.encoder_micro_batch_size
        if chunk_size <= 0 or images.shape[0] <= chunk_size:
            return self.encoder._encode_flat(images)
        return torch.cat(
            [self.encoder._encode_flat(chunk) for chunk in images.split(chunk_size, dim=0)],
            dim=0,
        )

    def _encode_views(self, visual: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        encoded = {}
        for name, images in visual.items():
            tokens = self._encode_flat_microbatched(images)
            if name in self.per_view_norm:
                tokens = self.per_view_norm[name](tokens)
            encoded[name] = tokens
        return encoded

    def encode_obs(self, obs: Mapping[str, object]) -> dict[str, torch.Tensor]:
        visual = obs["visual"]
        encoded = {}
        for name in self.source_view_names:
            images = visual[name]
            batch, time = images.shape[:2]
            tokens = self._encode_flat_microbatched(
                self._float_images(images.flatten(0, 1))
            ).unflatten(0, (batch, time))
            if name in self.per_view_norm:
                tokens = self.per_view_norm[name](tokens)
            encoded[name] = tokens
        visual_features = torch.cat([encoded[name] for name in self.source_view_names], dim=-1)
        return {
            "visual": visual_features,
            "proprio": self.proprio_encoder(obs["proprio"]),
        }

    def encode_act(self, actions: torch.Tensor) -> torch.Tensor:
        return self.action_encoder(actions)

    def _project_query(
        self,
        visual: Mapping[str, torch.Tensor],
        proprio: torch.Tensor,
    ) -> torch.Tensor:
        flat_visual = [visual[name].flatten(1) for name in self.source_view_names]
        state = self.state_projector(torch.cat([*flat_visual, proprio], dim=-1))
        return self.state_norm(state)

    def _project_future(
        self,
        visual: Mapping[str, torch.Tensor],
        proprio: torch.Tensor,
    ) -> torch.Tensor:
        # Shared-projector evaluation with unavailable source views set to zero.
        weight = self.state_projector.weight.index_select(1, self.future_projector_columns)
        flat_visual = [visual[name].flatten(1) for name in self.target_view_names]
        state = F.linear(torch.cat([*flat_visual, proprio], dim=-1), weight, self.state_projector.bias)
        return self.state_norm(state)

    @staticmethod
    def gather_future_keys(local_keys: torch.Tensor) -> tuple[torch.Tensor, int]:
        local_keys = local_keys.detach()
        if not dist.is_available() or not dist.is_initialized():
            return local_keys, 0
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        global_keys = torch.empty(
            (world_size * local_keys.shape[0], *local_keys.shape[1:]),
            dtype=local_keys.dtype,
            device=local_keys.device,
        )
        dist.all_gather_into_tensor(global_keys, local_keys.contiguous())
        return global_keys, rank * local_keys.shape[0]

    def info_nce(
        self,
        predictions: torch.Tensor,
        future_keys: torch.Tensor,
        positive_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type=predictions.device.type, enabled=False):
            logits = (predictions.float() @ self.W.float()) @ future_keys.float().T
            logits = logits - logits.amax(dim=1, keepdim=True)
            labels = torch.arange(predictions.shape[0], device=predictions.device) + int(positive_offset)
            return F.cross_entropy(logits, labels), logits

    def forward(
        self,
        obs: Mapping[str, object],
        act: torch.Tensor,
        global_future_keys: torch.Tensor | None = None,
        positive_offset: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
        visual = obs["visual"]
        query_images = self._augment_views(visual, self.source_view_names, time_index=0)
        future_images = self._augment_views(visual, self.target_view_names, time_index=self.num_hist)
        query_visual = self._encode_views(query_images)
        query_proprio = self.proprio_encoder(obs["proprio"][:, 0])
        query_state = self._project_query(query_visual, query_proprio)
        with torch.no_grad():
            future_visual = self._encode_views(future_images)
            future_proprio = self.proprio_encoder(obs["proprio"][:, self.num_hist])
            future_state = self._project_future(future_visual, future_proprio)

        if act.shape[-1] == self.action_encoder.in_chans:
            action_window = act[:, 0] if act.dim() >= 3 else act
        elif tuple(act.shape[-2:]) == (self.frameskip, self.action_encoder.action_dim):
            action_window = act[:, 0] if act.dim() >= 4 else act
        else:
            action_window = act[:, : self.frameskip]
        action_latent = self.encode_act(action_window)
        predictions = self.transition(torch.cat([query_state, action_latent], dim=-1))

        if global_future_keys is None:
            global_future_keys, inferred_offset = self.gather_future_keys(future_state)
            if positive_offset is None:
                positive_offset = inferred_offset
        else:
            global_future_keys = global_future_keys.detach()
        positive_offset = 0 if positive_offset is None else int(positive_offset)
        loss, logits = self.info_nce(predictions, global_future_keys, positive_offset)
        components: dict[str, torch.Tensor | int] = {
            "loss": loss,
            "taco_loss": loss,
            "logits": logits.detach(),
            "future_keys": future_state,
            "global_batch_size": int(global_future_keys.shape[0]),
            "negative_count": int(global_future_keys.shape[0] - 1),
        }
        return loss, components
