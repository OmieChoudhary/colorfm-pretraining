from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if pred.shape != target.shape or pred.shape != mask.shape:
        raise ValueError("pred, target, and mask must have identical shape")
    if mask.sum() == 0:
        return F.mse_loss(pred, target)
    return F.mse_loss(pred[mask], target[mask])


def lab_regression_loss(pred_lab: torch.Tensor, target_lab: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_lab, target_lab)


def spectral_smoothness_loss(reconstruction: torch.Tensor) -> torch.Tensor:
    """Penalize unrealistic wavelength-to-wavelength jumps.

    This is intentionally small and optional. It acts like a domain prior for reflectance curves.
    """
    if reconstruction.shape[1] < 3:
        return torch.tensor(0.0, device=reconstruction.device)
    first_diff = reconstruction[:, 1:] - reconstruction[:, :-1]
    second_diff = first_diff[:, 1:] - first_diff[:, :-1]
    return torch.mean(second_diff**2)
