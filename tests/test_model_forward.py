import torch

from colorfm.models.spectral_transformer import SpectralTransformer
from colorfm.objectives.losses import masked_reconstruction_loss


def test_model_forward_shapes():
    model = SpectralTransformer(seq_len=41, d_model=32, n_heads=4, n_layers=1, dim_feedforward=64)
    spectrum = torch.rand(2, 41)
    mask = torch.zeros(2, 41, dtype=torch.bool)
    mask[:, ::3] = True
    out = model(spectrum, mask)
    assert out["reconstruction"].shape == (2, 41)
    assert out["lab"].shape == (2, 3)
    assert out["embedding"].shape == (2, 32)


def test_masked_loss_is_scalar():
    pred = torch.rand(2, 41)
    target = torch.rand(2, 41)
    mask = torch.zeros(2, 41, dtype=torch.bool)
    mask[:, :5] = True
    loss = masked_reconstruction_loss(pred, target, mask)
    assert loss.ndim == 0
