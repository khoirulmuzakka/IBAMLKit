"""Reusable preprocessing and invertible transforms for training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


class ArrayTransform:
    """Base class for invertible array transforms."""

    def __init__(self) -> None:
        self.is_fitted = False
        self.input_dimension: int | None = None

    def fit(self, x: np.ndarray) -> "ArrayTransform":
        raise NotImplementedError

    def transform(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class IdentityTransform(ArrayTransform):
    """No-op transform."""

    def fit(self, x: np.ndarray) -> "IdentityTransform":
        x = np.asarray(x, dtype=np.float32)
        self.input_dimension = int(x.shape[1])
        self.is_fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float32)


class ConstantFactorTransform(ArrayTransform):
    """Multiply arrays by a constant factor."""

    def __init__(self, factor: float) -> None:
        super().__init__()
        self.factor = float(factor)
        if self.factor == 0.0:
            raise ValueError("factor must be non-zero.")

    def fit(self, x: np.ndarray) -> "ConstantFactorTransform":
        x = np.asarray(x, dtype=np.float32)
        self.input_dimension = int(x.shape[1])
        self.is_fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("ConstantFactorTransform must be fitted before transform().")
        return (np.asarray(x, dtype=np.float32) * self.factor).astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("ConstantFactorTransform must be fitted before inverse_transform().")
        return (np.asarray(x, dtype=np.float32) / self.factor).astype(np.float32)


class MinMaxScaler(ArrayTransform):
    """Column-wise min-max scaling."""

    def __init__(self, low: float = 0.0, high: float = 1.0) -> None:
        super().__init__()
        if high <= low:
            raise ValueError("high must be greater than low.")
        self.low = float(low)
        self.high = float(high)
        self.x_min: np.ndarray | None = None
        self.x_scale: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "MinMaxScaler":
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError("MinMaxScaler expects a 2D array.")
        self.input_dimension = int(x.shape[1])
        x_min = np.min(x, axis=0)
        x_max = np.max(x, axis=0)
        x_scale = x_max - x_min
        x_scale = np.where(x_scale > 0.0, x_scale, 1.0)
        self.x_min = x_min.astype(np.float32)
        self.x_scale = x_scale.astype(np.float32)
        self.is_fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.x_min is None or self.x_scale is None:
            raise RuntimeError("MinMaxScaler must be fitted before transform().")
        x = np.asarray(x, dtype=np.float32)
        return (self.low + (self.high - self.low) * (x - self.x_min) / self.x_scale).astype(
            np.float32
        )

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.x_min is None or self.x_scale is None:
            raise RuntimeError("MinMaxScaler must be fitted before inverse_transform().")
        x = np.asarray(x, dtype=np.float32)
        return (((x - self.low) / (self.high - self.low)) * self.x_scale + self.x_min).astype(
            np.float32
        )


class StandardScaler(ArrayTransform):
    """Column-wise zero-mean, unit-variance scaling."""

    def __init__(self) -> None:
        super().__init__()
        self.x_mean: np.ndarray | None = None
        self.x_std: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "StandardScaler":
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError("StandardScaler expects a 2D array.")
        self.input_dimension = int(x.shape[1])
        x_mean = np.mean(x, axis=0)
        x_std = np.std(x, axis=0)
        x_std = np.where(x_std > 0.0, x_std, 1.0)
        self.x_mean = x_mean.astype(np.float32)
        self.x_std = x_std.astype(np.float32)
        self.is_fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.x_mean is None or self.x_std is None:
            raise RuntimeError("StandardScaler must be fitted before transform().")
        x = np.asarray(x, dtype=np.float32)
        return ((x - self.x_mean) / self.x_std).astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.x_mean is None or self.x_std is None:
            raise RuntimeError("StandardScaler must be fitted before inverse_transform().")
        x = np.asarray(x, dtype=np.float32)
        return (x * self.x_std + self.x_mean).astype(np.float32)


class TransformPipeline(ArrayTransform):
    """Compose multiple invertible transforms."""

    def __init__(self, transforms: Sequence[ArrayTransform]) -> None:
        super().__init__()
        self.transforms = list(transforms)

    def fit(self, x: np.ndarray) -> "TransformPipeline":
        x_work = np.asarray(x, dtype=np.float32)
        if x_work.ndim != 2:
            raise ValueError("TransformPipeline expects a 2D array.")
        self.input_dimension = int(x_work.shape[1])
        for transform in self.transforms:
            x_work = transform.fit_transform(x_work)
        self.is_fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("TransformPipeline must be fitted before transform().")
        x_work = np.asarray(x, dtype=np.float32)
        for transform in self.transforms:
            x_work = transform.transform(x_work)
        return x_work.astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("TransformPipeline must be fitted before inverse_transform().")
        x_work = np.asarray(x, dtype=np.float32)
        for transform in reversed(self.transforms):
            x_work = transform.inverse_transform(x_work)
        return x_work.astype(np.float32)


class SpectrumPreprocessor:
    """Base class for non-invertible spectrum preprocessors."""

    def fit(self, x: np.ndarray) -> "SpectrumPreprocessor":
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.transform(x)


class BinMerger(SpectrumPreprocessor):
    """Merge adjacent bins by summation."""

    def __init__(self, bins_per_group: int = 2) -> None:
        if bins_per_group < 1:
            raise ValueError("bins_per_group must be >= 1.")
        self.bins_per_group = int(bins_per_group)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError("BinMerger expects a 2D array.")
        n_rows, n_cols = x.shape
        padded_cols = ((n_cols + self.bins_per_group - 1) // self.bins_per_group) * self.bins_per_group
        if padded_cols != n_cols:
            padded = np.zeros((n_rows, padded_cols), dtype=np.float32)
            padded[:, :n_cols] = x
            x = padded
        return x.reshape(n_rows, -1, self.bins_per_group).sum(axis=2).astype(np.float32)


class ROISelector(SpectrumPreprocessor):
    """Select and concatenate channel regions of interest."""

    def __init__(self, rois: Sequence[tuple[int, int]]) -> None:
        self.rois = [(int(start), int(stop)) for start, stop in rois]

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError("ROISelector expects a 2D array.")
        parts = [x[:, start:stop] for start, stop in self.rois]
        if not parts:
            return np.zeros((x.shape[0], 0), dtype=np.float32)
        return np.concatenate(parts, axis=1).astype(np.float32)


class SpectrumPreprocessorPipeline(SpectrumPreprocessor):
    """Compose multiple spectrum preprocessors."""

    def __init__(self, preprocessors: Sequence[SpectrumPreprocessor]) -> None:
        self.preprocessors = list(preprocessors)

    def fit(self, x: np.ndarray) -> "SpectrumPreprocessorPipeline":
        x_work = np.asarray(x, dtype=np.float32)
        for preprocessor in self.preprocessors:
            x_work = preprocessor.fit_transform(x_work)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x_work = np.asarray(x, dtype=np.float32)
        for preprocessor in self.preprocessors:
            x_work = preprocessor.transform(x_work)
        return x_work.astype(np.float32)


@dataclass(frozen=True)
class TrainValTestSplit:
    """Array split container for supervised learning."""

    train_inputs: np.ndarray
    train_targets: np.ndarray
    val_inputs: np.ndarray
    val_targets: np.ndarray
    test_inputs: np.ndarray
    test_targets: np.ndarray
    test_target_lengths: np.ndarray | None = None


def shuffle_in_unison(
    inputs: np.ndarray,
    targets: np.ndarray,
    *,
    seed: int = 0,
    target_lengths: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Shuffle aligned arrays using one shared permutation."""

    inputs = np.asarray(inputs)
    targets = np.asarray(targets)
    if inputs.shape[0] != targets.shape[0]:
        raise ValueError("inputs and targets must have the same sample count.")
    if target_lengths is not None and np.asarray(target_lengths).shape[0] != inputs.shape[0]:
        raise ValueError("target_lengths must match the sample count.")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(inputs.shape[0])
    inputs = np.asarray(inputs[perm], dtype=np.float32)
    targets = np.asarray(targets[perm], dtype=np.float32)
    if target_lengths is None:
        return inputs, targets, None
    return inputs, targets, np.asarray(target_lengths[perm], dtype=np.int32)


def split_train_val_test(
    inputs: np.ndarray,
    targets: np.ndarray,
    *,
    val_count: int,
    test_count: int,
    target_lengths: np.ndarray | None = None,
) -> TrainValTestSplit:
    """Split arrays into train/validation/test blocks."""

    inputs = np.asarray(inputs, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    if inputs.shape[0] != targets.shape[0]:
        raise ValueError("inputs and targets must have the same sample count.")
    sample_count = int(inputs.shape[0])
    train_count = sample_count - int(val_count) - int(test_count)
    if train_count <= 0:
        raise ValueError("Not enough samples for the requested train/val/test split.")
    test_lengths_out = None
    if target_lengths is not None:
        target_lengths = np.asarray(target_lengths, dtype=np.int32)
        if target_lengths.shape[0] != sample_count:
            raise ValueError("target_lengths must match the sample count.")
        test_lengths_out = target_lengths[train_count + val_count :]
    return TrainValTestSplit(
        train_inputs=inputs[:train_count],
        train_targets=targets[:train_count],
        val_inputs=inputs[train_count : train_count + val_count],
        val_targets=targets[train_count : train_count + val_count],
        test_inputs=inputs[train_count + val_count :],
        test_targets=targets[train_count + val_count :],
        test_target_lengths=test_lengths_out,
    )
