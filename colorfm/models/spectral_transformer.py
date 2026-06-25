from __future__ import annotations

import torch
from torch import nn


class SpectralTransformer(nn.Module):
    """Tiny transformer for masked spectral modeling.

    Input shape: (batch, n_wavelengths). Each scalar reflectance value is embedded as a token
    and combined with a learned positional embedding.
    """

    def __init__(
        self,
        seq_len: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.value_embed = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.reconstruct_head = nn.Linear(d_model, 1)
        self.lab_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),
        )
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, spectrum: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if spectrum.ndim != 2:
            raise ValueError(f"Expected spectrum shape (batch, seq_len), got {tuple(spectrum.shape)}")
        if spectrum.shape[1] != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {spectrum.shape[1]}")
        x = spectrum.unsqueeze(-1)
        if mask is not None:
            # Replace masked reflectance values with zero. The model must infer them from context.
            x = x.masked_fill(mask.unsqueeze(-1), 0.0)
        h = self.value_embed(x) + self.pos_embed
        h = self.encoder(h)
        reconstruction = self.reconstruct_head(h).squeeze(-1)
        pooled = h.mean(dim=1)
        lab = self.lab_head(pooled)
        return {"reconstruction": reconstruction, "lab": lab, "embedding": pooled}
