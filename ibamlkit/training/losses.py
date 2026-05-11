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


class PeakAwareLoss(nn.Module):
    """Log-space loss with local gradient matching for sharper peaks."""

    def __init__(
        self,
        *,
        amplitude_weight: float = 1.0,
        gradient_weight: float = 0.25,
        curvature_weight: float = 0.05,
    ) -> None:
        super().__init__()
        self.amplitude_weight = float(amplitude_weight)
        self.gradient_weight = float(gradient_weight)
        self.curvature_weight = float(curvature_weight)

    @staticmethod
    def _gradient(values: torch.Tensor) -> torch.Tensor:
        return values[:, 1:] - values[:, :-1]

    @staticmethod
    def _curvature(values: torch.Tensor) -> torch.Tensor:
        return values[:, 2:] - 2.0 * values[:, 1:-1] + values[:, :-2]

    def forward(self, targets: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
        targets = torch.clamp(targets, min=0.0)
        predictions = torch.clamp(predictions, min=0.0)
        log_targets = torch.log1p(targets)
        log_predictions = torch.log1p(predictions)

        amplitude_loss = torch.mean((log_predictions - log_targets) ** 2)
        gradient_loss = torch.mean(
            (self._gradient(log_predictions) - self._gradient(log_targets)) ** 2
        )
        curvature_loss = torch.mean(
            (self._curvature(log_predictions) - self._curvature(log_targets)) ** 2
        )
        return (
            self.amplitude_weight * amplitude_loss
            + self.gradient_weight * gradient_loss
            + self.curvature_weight * curvature_loss
        )
