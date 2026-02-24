import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models import resnet18, ResNet18_Weights
except Exception:
    resnet18 = None
    ResNet18_Weights = None


def _build_resnet18_backbone(
    pretrained: bool = True,
    local_weights_path: str = ""
) -> nn.Module:
    m = resnet18(weights=None)

    if pretrained:
        if local_weights_path and os.path.exists(local_weights_path):
            state_dict = torch.load(local_weights_path, map_location="cpu")
            m.load_state_dict(state_dict)
            print(f"Loaded local ResNet18 weights from {local_weights_path}")
        else:
            raise FileNotFoundError(
                f"Pretrained=True but local_weights_path not found: {local_weights_path}"
            )

    m.fc = nn.Identity()
    return m


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, depth: int, dropout: float) -> None:
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(d, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class ClassifierConfig:
    use_images: bool = True
    use_state_action: bool = True
    image_pretrained: bool = True

    img_embed_dim: int = 256
    sa_embed_dim: int = 256
    fusion_hidden: int = 512
    fusion_depth: int = 3
    dropout: float = 0.1
    resnet_path: str = "/home/gy/Documents/DAgger/robosuite/RL/models/resnet18-f37072fd.pth"


class PRVMClassifier(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, cfg: ClassifierConfig) -> None:
        super().__init__()
        assert cfg.use_images or cfg.use_state_action

        self.cfg = cfg
        self.state_dim = state_dim
        self.action_dim = action_dim

        if cfg.use_images:
            self.img_backbone_agent = _build_resnet18_backbone(
                pretrained=cfg.image_pretrained,
                local_weights_path=cfg.resnet_path,
            )
            self.img_backbone_wrist = _build_resnet18_backbone(
                pretrained=cfg.image_pretrained,
                local_weights_path=cfg.resnet_path,
            )
            self.img_proj = nn.Sequential(
                nn.Linear(512 * 2, cfg.img_embed_dim),
                nn.SELU(inplace=True),
                nn.Dropout(cfg.dropout),
            )
        else:
            self.img_backbone_agent = None
            self.img_backbone_wrist = None
            self.img_proj = None

        if cfg.use_state_action:
            self.sa_mlp = MLP(
                in_dim=state_dim + action_dim,
                hidden=cfg.fusion_hidden,
                out_dim=cfg.sa_embed_dim,
                depth=3,
                dropout=cfg.dropout,
            )
        else:
            self.sa_mlp = None

        in_fusion = 0
        if cfg.use_images:
            in_fusion += cfg.img_embed_dim
        if cfg.use_state_action:
            in_fusion += cfg.sa_embed_dim

        self.fusion = MLP(
            in_dim=in_fusion,
            hidden=cfg.fusion_hidden,
            out_dim=1,
            depth=cfg.fusion_depth,
            dropout=cfg.dropout,
        )

    @staticmethod
    def _normalize_uint8_images(x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        return x

    def forward(
        self,
        img_agent: Optional[torch.Tensor] = None,
        img_wrist: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feats = []

        if self.cfg.use_images:
            assert img_agent is not None and img_wrist is not None
            xa = self._normalize_uint8_images(img_agent)
            xw = self._normalize_uint8_images(img_wrist)
            fa = self.img_backbone_agent(xa)
            fw = self.img_backbone_wrist(xw)
            f = torch.cat([fa, fw], dim=-1)
            f = self.img_proj(f)
            feats.append(f)

        if self.cfg.use_state_action:
            assert state is not None and action is not None
            sa = torch.cat([state, action], dim=-1)
            sa_feat = self.sa_mlp(sa)
            feats.append(sa_feat)

        x = torch.cat(feats, dim=-1) if len(feats) > 1 else feats[0]
        logit = self.fusion(x).squeeze(-1)
        return logit
