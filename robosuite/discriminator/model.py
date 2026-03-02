import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

try:
    from torchvision.models import resnet18
except Exception:
    resnet18 = None


def _build_resnet18_backbone(pretrained: bool = True, local_weights_path: str = "") -> nn.Module:
    m = resnet18(weights=None)
    if pretrained and local_weights_path and os.path.exists(local_weights_path):
        state_dict = torch.load(local_weights_path, map_location="cpu")
        m.load_state_dict(state_dict)
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
class DetectorConfig:
    use_images: bool = True
    img_embed_dim: int = 128
    proprio_dim: int = 32
    action_dim: int = 7
    temporal_hidden: int = 256
    fusion_hidden: int = 256
    resnet_path: str = "/home/gy/Documents/DAgger/robosuite/RL/models/resnet18-f37072fd.pth"


class SequenceFailureDetector(nn.Module):
    def __init__(self, cfg: DetectorConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Spatial Encoders
        if cfg.use_images:
            self.img_backbone = _build_resnet18_backbone(True, cfg.resnet_path)
            self.img_proj = nn.Linear(512, cfg.img_embed_dim)
            step_feat_dim = cfg.img_embed_dim + cfg.proprio_dim
        else:
            step_feat_dim = cfg.proprio_dim

        # Temporal Encoder for History + Current State
        self.temporal_encoder = nn.GRU(
            input_size=step_feat_dim, 
            hidden_size=cfg.temporal_hidden, 
            batch_first=True
        )

        # Fusion network: Temporal Context + Current Action -> Logit
        self.fusion = MLP(
            in_dim=cfg.temporal_hidden + cfg.action_dim,
            hidden=cfg.fusion_hidden,
            out_dim=1,
            depth=3,
            dropout=0.1
        )

    def forward(
        self, 
        img_agent_seq: Optional[torch.Tensor] = None, # Shape: [B, T, C, H, W]
        proprio_seq: torch.Tensor = None,             # Shape: [B, T, D_p]
        current_action: torch.Tensor = None           # Shape: [B, D_a]
    ) -> torch.Tensor:
        
        B, T = proprio_seq.shape[:2]
        step_features = []

        if self.cfg.use_images:
            # Flatten B and T to process all images in a single forward pass
            ia_flat = img_agent_seq.view(B * T, *img_agent_seq.shape[2:]).float() / 255.0
            fa = self.img_proj(self.img_backbone(ia_flat))

            fa = fa.view(B, T, -1)
            step_features.extend([fa])

        step_features.append(proprio_seq)
        
        # Concat spatial features for each timestep
        x_seq = torch.cat(step_features, dim=-1) # Shape: [B, T, step_feat_dim]

        # Temporal encoding
        _, h_n = self.temporal_encoder(x_seq)
        context = h_n.squeeze(0) # Shape: [B, temporal_hidden]

        # Fuse with current action
        fusion_in = torch.cat([context, current_action], dim=-1)
        logit = self.fusion(fusion_in).squeeze(-1) # Shape: [B]
        
        return logit