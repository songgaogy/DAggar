import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


def sinusoidal_time_embedding(t: torch.Tensor, dim: int):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(0, half, device=t.device, dtype=torch.float32) / half
    )
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ImageEncoder(nn.Module):
    def __init__(self, out_dim: int = 256, pretrained: bool = True, freeze_backbone: bool = False, 
                 path: str = "/home/dodo/Documents/DAggar/robosuite/robosuite/RL/models/resnet18-f37072fd.pth"):
        super().__init__()
        net = resnet18(weights=None)
        if pretrained:
            net.load_state_dict(torch.load(path, map_location="cpu"))
        net.fc = nn.Identity()
        self.backbone = net
        self.proj = nn.Linear(512, out_dim)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        h = self.backbone(x)
        h = self.proj(h)
        return h


class AdaLN(nn.Module):
    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.to_scale_shift = nn.Sequential(
            nn.Linear(cond_dim, 2 * hidden_dim),
        )

    def forward(self, x, cond):
        x = self.ln(x)
        ss = self.to_scale_shift(cond)
        scale, shift = ss.chunk(2, dim=-1)
        # Broadcast conditioning over token / sequence dimensions when x is [B, N, D].
        while scale.ndim < x.ndim:
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        return x * (1.0 + scale) + shift


class AdaLNBlock(nn.Module):
    def __init__(self, hidden_dim: int, cond_dim: int, dropout: float = 0.0):
        super().__init__()
        self.adaln = AdaLN(hidden_dim, cond_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cond):
        h = self.adaln(x, cond)
        h = F.gelu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        return x + h


class TemporalEncoder(nn.Module):
    def __init__(self, token_dim: int, n_layers: int = 2, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=n_heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, tokens):
        return self.enc(tokens)


class FlowPolicy(nn.Module):
    def __init__(
        self,
        act_dim: int,
        proprio_in_dim: int,
        img_dim: int = 256,
        prop_dim: int = 256,
        time_dim: int = 128,
        token_dim: int = 256,
        temporal_layers: int = 2,
        temporal_heads: int = 8,
        vel_hidden: int = 512,
        vel_layers: int = 6,
        dropout: float = 0.0,
        history_len: int = 1,
        action_chunk_size: int = 1,
        action_temporal_layers: int = 1,
        action_temporal_heads: int = 4,
        pretrained_resnet: bool = True,
        freeze_resnet: bool = False,
    ):
        super().__init__()
        self.act_dim = int(act_dim)
        self.time_dim = int(time_dim)
        self.chunk_size = int(action_chunk_size)
        if self.chunk_size <= 0:
            raise ValueError(f"action_chunk_size must be > 0, got {self.chunk_size}")
        if self.act_dim % self.chunk_size != 0:
            raise ValueError(
                f"act_dim ({self.act_dim}) must be divisible by action_chunk_size ({self.chunk_size})"
            )
        self.step_act_dim = self.act_dim // self.chunk_size

        self.img_enc = ImageEncoder(out_dim=img_dim, pretrained=pretrained_resnet, freeze_backbone=freeze_resnet)
        self.prop_enc = nn.Sequential(
            nn.Linear(proprio_in_dim, prop_dim),
            nn.GELU(),
            nn.Linear(prop_dim, prop_dim),
            nn.GELU(),
        )

        self.img_to_token = nn.Linear(img_dim, token_dim)
        self.prop_to_token = nn.Linear(prop_dim, token_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))

        self.temporal = TemporalEncoder(token_dim=token_dim, n_layers=temporal_layers, n_heads=temporal_heads, dropout=dropout)

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )

        self.cond_ln = nn.LayerNorm(token_dim)
        self.token_time_adaln = AdaLN(hidden_dim=token_dim, cond_dim=token_dim)
        self.time_token = nn.Parameter(torch.zeros(1, 1, token_dim))

        self.in_proj = nn.Linear(self.step_act_dim, vel_hidden)
        self.cond_proj = nn.Linear(token_dim, vel_hidden)

        # 1 time token + 1 cls token + history_len image tokens + 1 proprio token
        self.pos_emb = nn.Parameter(torch.zeros(1, history_len + 3, token_dim))
        self.action_pos_emb = nn.Parameter(torch.zeros(1, self.chunk_size, vel_hidden))
        if action_temporal_layers > 0:
            self.action_temporal = TemporalEncoder(
                token_dim=vel_hidden,
                n_layers=action_temporal_layers,
                n_heads=action_temporal_heads,
                dropout=dropout,
            )
        else:
            self.action_temporal = nn.Identity()

        blocks = []
        for _ in range(vel_layers):
            blocks.append(AdaLNBlock(hidden_dim=vel_hidden, cond_dim=vel_hidden, dropout=dropout))
        self.blocks = nn.ModuleList(blocks)

        self.out_ln = nn.LayerNorm(vel_hidden)
        self.out = nn.Linear(vel_hidden, self.step_act_dim)

        self.gate = nn.Sequential(
            nn.Linear(vel_hidden, vel_hidden),
            nn.GELU(),
            nn.Linear(vel_hidden, self.step_act_dim),
            nn.Tanh(),
        )

    def forward(self, x_t, t, images, proprio):
        B, K = images.shape[0], images.shape[1]

        imgs = images.reshape(B * K, images.shape[2], images.shape[3], images.shape[4])
        img_feat = self.img_enc(imgs).reshape(B, K, -1)

        prop_feat = self.prop_enc(proprio)

        img_tok = self.img_to_token(img_feat)
        prop_tok = self.prop_to_token(prop_feat).unsqueeze(1)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens_base = torch.cat([cls_tokens, img_tok, prop_tok], dim=1)

        t_emb = sinusoidal_time_embedding(t, self.time_dim)
        t_cond = self.time_mlp(t_emb)
        t_tok = self.time_token + t_cond.unsqueeze(1)

        tokens = torch.cat([t_tok, tokens_base], dim=1)
        tokens = tokens + self.pos_emb[:, :tokens.shape[1], :]
        tokens = self.token_time_adaln(tokens, t_cond)

        tokens = self.temporal(tokens) 
        cond = tokens[:, 1]  # cls token follows the time token
        cond = self.cond_ln(cond) 
        cond_h = self.cond_proj(cond) 

        x_tokens = x_t.view(B, self.chunk_size, self.step_act_dim)
        h = self.in_proj(x_tokens)
        for blk in self.blocks:
            h = blk(h, cond_h)

        h = h + self.action_pos_emb
        h = self.action_temporal(h)
        h = self.out_ln(h)
        v = self.out(h) 
        v = v * (1.0 + 0.5 * self.gate(h))
        return v.reshape(B, self.act_dim)

    def get_cond_features(self, images, proprio):
        B, K = images.shape[0], images.shape[1]
        imgs = images.reshape(B * K, images.shape[2], images.shape[3], images.shape[4])
        img_feat = self.img_enc(imgs).reshape(B, K, -1)
        prop_feat = self.prop_enc(proprio)
        
        img_tok = self.img_to_token(img_feat)
        prop_tok = self.prop_to_token(prop_feat).unsqueeze(1)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        tokens_base = torch.cat([cls_tokens, img_tok, prop_tok], dim=1)
        return tokens_base

    def forward_with_features(self, x_t, t, tokens_base):
        B = x_t.shape[0]
        t_emb = sinusoidal_time_embedding(t, self.time_dim)
        t_cond = self.time_mlp(t_emb)
        t_tok = self.time_token + t_cond.unsqueeze(1)
        tokens = torch.cat([t_tok, tokens_base], dim=1)
        tokens = tokens + self.pos_emb[:, :tokens.shape[1], :]
        tokens = self.token_time_adaln(tokens, t_cond)
        
        tokens = self.temporal(tokens)
        cond = tokens[:, 1]
        cond = self.cond_ln(cond)
        cond_h = self.cond_proj(cond)

        x_tokens = x_t.view(B, self.chunk_size, self.step_act_dim)
        h = self.in_proj(x_tokens)
        for blk in self.blocks:
            h = blk(h, cond_h)

        h = h + self.action_pos_emb
        h = self.action_temporal(h)
        h = self.out_ln(h)
        v = self.out(h)
        v = v * (1.0 + 0.5 * self.gate(h))
        return v.reshape(B, self.act_dim)
