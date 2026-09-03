import torch.nn as nn


class MLPEmbedding(nn.Module):
    def __init__(
        self,
        in_chans=8,
        emb_dim=64,
        hidden_dim=128,
        dropout=0.0,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(in_chans, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, x):
        return self.net(x)
