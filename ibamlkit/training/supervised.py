"""Generic supervised trainers for IBAMLKit models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
import copy

import torch
from torch.utils.data import DataLoader, TensorDataset

from ibamlkit.training.base import ModelTrainer, TrainingResult, TrainableTensorModel
from ibamlkit.training.losses import Chi2Loss


@dataclass(frozen=True)
class EpochSchedule:
    """One optimizer phase in a supervised training run."""

    learning_rate: float
    epochs: int
    batch_size: int


class SupervisedTrainer(ModelTrainer):
    """Basic trainer for supervised forward-model fitting."""

    def __init__(
        self,
        *,
        loss_fn: torch.nn.Module | None = None,
        optimizer_name: str = "adamw",
        weight_decay: float = 1e-3,
        max_grad_norm: float | None = 1.0,
        early_stopping_patience: int = 10,
        eval_batch_size: int = 500,
        device: str = "cpu",
        verbose: bool = True,
        log_every_epochs: int = 1,
    ) -> None:
        self.loss_fn = loss_fn or Chi2Loss()
        self.optimizer_name = optimizer_name.lower()
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.early_stopping_patience = early_stopping_patience
        self.eval_batch_size = eval_batch_size
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.log_every_epochs = max(1, int(log_every_epochs))

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _make_optimizer(self, model: TrainableTensorModel, learning_rate: float) -> torch.optim.Optimizer:
        if self.optimizer_name == "adam":
            return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=self.weight_decay)
        if self.optimizer_name == "sgd":
            return torch.optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=0.5,
                weight_decay=self.weight_decay,
            )
        if self.optimizer_name == "adamw":
            return torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                betas=(0.9, 0.999),
                weight_decay=self.weight_decay,
            )
        raise ValueError(f"Unsupported optimizer_name: {self.optimizer_name}")

    def _loss_over_loader(self, model: TrainableTensorModel, loader: DataLoader) -> float:
        model.eval()
        total_loss = 0.0
        total_items = 0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                predictions = model(inputs)
                batch_size = inputs.shape[0]
                total_loss += self.loss_fn(targets, predictions).detach().cpu().item() * batch_size
                total_items += batch_size
        return total_loss / max(total_items, 1)

    def fit(
        self,
        model: TrainableTensorModel,
        /,
        *,
        train_inputs: torch.Tensor,
        train_targets: torch.Tensor,
        val_inputs: torch.Tensor,
        val_targets: torch.Tensor,
        schedule: Iterable[EpochSchedule] = (EpochSchedule(learning_rate=1e-3, epochs=300, batch_size=64),),
    ) -> TrainingResult:
        train_inputs = model.transform_inputs(train_inputs)
        val_inputs = model.transform_inputs(val_inputs)
        train_targets = train_targets if isinstance(train_targets, torch.Tensor) else torch.as_tensor(train_targets, dtype=torch.float32)
        val_targets = val_targets if isinstance(val_targets, torch.Tensor) else torch.as_tensor(val_targets, dtype=torch.float32)
        train_targets = train_targets.to(dtype=torch.float32)
        val_targets = val_targets.to(dtype=torch.float32)

        model.validate_input_shape(train_inputs)
        model.validate_input_shape(val_inputs)

        model.to(self.device)
        pin_memory = self.device.type == "cuda"
        train_dataset = TensorDataset(train_inputs, train_targets)
        eval_train_loader = DataLoader(train_dataset, batch_size=self.eval_batch_size, shuffle=False, pin_memory=pin_memory)
        val_loader = DataLoader(
            TensorDataset(val_inputs, val_targets),
            batch_size=self.eval_batch_size,
            shuffle=False,
            pin_memory=pin_memory,
        )

        self._log(
            "Training started: "
            f"device={self.device.type}, "
            f"train_samples={len(train_dataset)}, "
            f"val_samples={len(val_targets)}, "
            f"input_dim={train_inputs.shape[1]}, "
            f"target_dim={train_targets.shape[1]}"
        )
        best_state = copy.deepcopy(model.state_dict())
        best_val_loss = self._loss_over_loader(model, val_loader)
        train_loss = self._loss_over_loader(model, eval_train_loader)
        epochs_completed = 0
        self._log(
            f"Initial losses: train={train_loss:.6f}, val={best_val_loss:.6f}"
        )

        schedule_list = list(schedule)
        for phase_index, phase in enumerate(schedule_list, start=1):
            self._log(
                "Phase "
                f"{phase_index}/{len(schedule_list)}: "
                f"lr={phase.learning_rate:g}, epochs={phase.epochs}, batch_size={phase.batch_size}"
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=phase.batch_size,
                shuffle=True,
                pin_memory=pin_memory,
            )
            optimizer = self._make_optimizer(model, phase.learning_rate)
            stale_epochs = 0

            for phase_epoch in range(1, phase.epochs + 1):
                model.train()
                for inputs, targets in train_loader:
                    inputs = inputs.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)
                    predictions = model(inputs)
                    loss = self.loss_fn(targets, predictions)
                    optimizer.zero_grad()
                    loss.backward()
                    if self.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.max_grad_norm)
                    optimizer.step()

                epochs_completed += 1
                val_loss = self._loss_over_loader(model, val_loader)
                improved = val_loss <= best_val_loss
                if val_loss <= best_val_loss:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(model.state_dict())
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                should_log = (
                    phase_epoch == 1
                    or phase_epoch % self.log_every_epochs == 0
                    or phase_epoch == phase.epochs
                    or improved
                )
                if should_log:
                    train_loss_epoch = self._loss_over_loader(model, eval_train_loader)
                    status = "improved" if improved else f"stale={stale_epochs}"
                    self._log(
                        f"  epoch {phase_epoch}/{phase.epochs} "
                        f"(global {epochs_completed}): "
                        f"train={train_loss_epoch:.6f}, "
                        f"val={val_loss:.6f}, "
                        f"best={best_val_loss:.6f}, "
                        f"{status}"
                    )
                if stale_epochs >= self.early_stopping_patience:
                    self._log(
                        "  early stopping triggered: "
                        f"no val improvement for {self.early_stopping_patience} epoch(s)"
                    )
                    break

        model.load_state_dict(best_state)
        train_loss = self._loss_over_loader(model, eval_train_loader)
        model.eval()
        self._log(
            f"Training finished: epochs_completed={epochs_completed}, "
            f"train={train_loss:.6f}, val={best_val_loss:.6f}"
        )

        return TrainingResult(
            epochs_completed=epochs_completed,
            metrics={
                "train_loss": float(train_loss),
                "val_loss": float(best_val_loss),
            },
            metadata={
                "device": self.device.type,
                "optimizer": self.optimizer_name,
            },
        )
