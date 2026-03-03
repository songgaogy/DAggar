from dataclasses import dataclass

import torch
import torch.nn as nn

from robosuite.policy.flow import FlowPolicy, sinusoidal_time_embedding


@dataclass
class BackboneBuildResult:
    backbone: FlowPolicy
    history_len: int
    token_dim: int
    proprio_in_dim: int
    loaded_params: int


def _infer_num_transformer_layers(state_dict: dict[str, torch.Tensor], prefix: str) -> int:
    layer_ids = set()
    needle = prefix + ".enc.layers."
    for key in state_dict.keys():
        if key.startswith(needle):
            rest = key[len(needle) :]
            layer_ids.add(int(rest.split(".")[0]))
    return (max(layer_ids) + 1) if layer_ids else 1


def _choose_heads(token_dim: int) -> int:
    for h in (8, 4, 2, 1):
        if token_dim % h == 0:
            return h
    return 1


def infer_backbone_shapes(state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
    img_dim = int(state_dict["img_enc.proj.weight"].shape[0])
    token_dim = int(state_dict["img_to_token.weight"].shape[0])
    prop_dim = int(state_dict["prop_enc.0.weight"].shape[0])
    proprio_in_dim = int(state_dict["prop_enc.0.weight"].shape[1])
    time_dim = int(state_dict["time_mlp.0.weight"].shape[1])
    history_len = int(state_dict["pos_emb"].shape[1]) - 3

    return {
        "img_dim": img_dim,
        "token_dim": token_dim,
        "prop_dim": prop_dim,
        "proprio_in_dim": proprio_in_dim,
        "time_dim": time_dim,
        "history_len": history_len,
    }


def build_policy_backbone_from_ckpt(
    ckpt_path: str,
    history_len: int = -1,
    freeze_backbone: bool = True,
    device: str = "cpu",
) -> BackboneBuildResult:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("ema_model", ckpt["model"])

    shapes = infer_backbone_shapes(state_dict)
    resolved_history = shapes["history_len"] if int(history_len) <= 0 else int(history_len)

    temporal_layers = _infer_num_transformer_layers(state_dict, "temporal")
    temporal_heads = _choose_heads(shapes["token_dim"])

    backbone = FlowPolicy(
        act_dim=1,
        proprio_in_dim=shapes["proprio_in_dim"],
        img_dim=shapes["img_dim"],
        prop_dim=shapes["prop_dim"],
        time_dim=shapes["time_dim"],
        token_dim=shapes["token_dim"],
        temporal_layers=temporal_layers,
        temporal_heads=temporal_heads,
        vel_hidden=64,
        vel_layers=1,
        history_len=resolved_history,
        action_chunk_size=1,
        action_temporal_layers=0,
        action_temporal_heads=1,
        pretrained_resnet=False,
        freeze_resnet=freeze_backbone,
    )

    load_prefixes = (
        "img_enc.",
        "prop_enc.",
        "img_to_token.",
        "prop_to_token.",
        "cls_token",
        "temporal.",
        "time_mlp.",
        "token_time_adaln.",
        "time_token",
        "cond_ln.",
        "pos_emb",
    )

    own = backbone.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    for key, val in state_dict.items():
        if not key.startswith(load_prefixes):
            continue
        if key not in own:
            continue
        if own[key].shape != val.shape:
            continue
        filtered[key] = val

    own.update(filtered)
    backbone.load_state_dict(own, strict=False)

    if freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad = False

    backbone.to(device)
    return BackboneBuildResult(
        backbone=backbone,
        history_len=resolved_history,
        token_dim=shapes["token_dim"],
        proprio_in_dim=shapes["proprio_in_dim"],
        loaded_params=len(filtered),
    )


class PolicyConditionEncoder(nn.Module):
    def __init__(self, backbone: FlowPolicy):
        super().__init__()
        self.backbone = backbone

    @property
    def output_dim(self) -> int:
        return int(self.backbone.cond_ln.normalized_shape[0])

    def forward(self, images: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        tokens_base = self.backbone.get_cond_features(images=images, proprio=proprio)
        batch_size = int(tokens_base.shape[0])

        t = torch.zeros(batch_size, device=images.device, dtype=images.dtype)
        t_emb = sinusoidal_time_embedding(t, self.backbone.time_dim)
        t_cond = self.backbone.time_mlp(t_emb)
        t_tok = self.backbone.time_token + t_cond.unsqueeze(1)

        tokens = torch.cat([t_tok, tokens_base], dim=1)
        tokens = tokens + self.backbone.pos_emb[:, : tokens.shape[1], :]
        tokens = self.backbone.token_time_adaln(tokens, t_cond)
        tokens = self.backbone.temporal(tokens)

        cond = tokens[:, 1]
        cond = self.backbone.cond_ln(cond)
        return cond


class BCEVisitationDiscriminator(nn.Module):
    def __init__(self, encoder: PolicyConditionEncoder, action_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        feat_dim = int(encoder.output_dim)

        self.head = nn.Sequential(
            nn.Linear(feat_dim + int(action_dim), hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, images: torch.Tensor, proprio: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(images=images, proprio=proprio)
        x = torch.cat([feat, actions], dim=-1)
        logits = self.head(x).squeeze(-1)
        return logits
