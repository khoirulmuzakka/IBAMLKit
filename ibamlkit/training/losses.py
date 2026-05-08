"""Training losses used by IBAMLKit trainers."""

from __future__ import annotations

import torch
from torch import nn


class Chi2Loss(nn.Module):
    """Mean chi-squared style loss used by legacy surrogate training."""

    def forward(self, targets: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
        return torch.mean((predictions - targets) ** 2 / (targets + 1.0))


class Log1pMSELoss(nn.Module):
    """Mean squared error in log-space for non-negative spectra."""

    def forward(self, targets: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
        targets = torch.clamp(targets, min=0.0)
        predictions = torch.clamp(predictions, min=0.0)
        return torch.mean((torch.log1p(predictions) - torch.log1p(targets)) ** 2)
