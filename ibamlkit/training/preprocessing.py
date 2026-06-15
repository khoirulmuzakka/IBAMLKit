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

    def transform(self, x: np.ndarray, *, out: np.ndarray | None = None) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(self, x: np.ndarray, *, out: np.ndarray | None = None) -> np.ndarray:
        self.fit(x)
        return self.transform(x, out=out)

    def inverse_transform(self, x: np.ndarray, *, out: np.ndarray | None = None) -> np.ndarray:
        raise NotImplementedError


class IdentityTransform(ArrayTransform):
    """No-op transform."""

    def fit(self, x: np.ndarray) -> "IdentityTransform":
        x = np.asarray(x, dtype=np.float32)
        self.input_dimension = int(x.shape[1])
        self.is_fitted = True
        return self

    def transform(self, x: np.ndarray, *, out: np.ndarray | None = None) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if out is None:
            return x
        np.copyto(out, x, casting="unsafe")
        return out

    def inverse_transform(self, x: np.ndarray, *, out: np.ndarray | None = None) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if out is None:
            return x
        np.copyto(out, x, casting="unsafe")
        return out


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

    def transform(self, x: np.ndarray, *, out: np.ndarray | None = None) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("ConstantFactorTransform must be fitted before transform().")
        x = np.asarray(x, dtype=np.float32)
        if out is None:
            return (x * self.factor).astype(np.float32)
        np.multiply(x, self.factor, out=out, casting="unsafe")
        return out

    def inverse_transform(self, x: np.ndarray, *, out: np.ndarray | None = None) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("ConstantFactorTransform must be fitted before inverse_transform().")
        x = np.asarray(x, dtype=np.float32)
        if out is None:
            return (x / self.factor).astype(np.float32)
        np.divide(x, self.factor, out=out, casting="unsafe")
        return out


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

    def transform(self, x: np.ndarray, *, out: np.ndarray | None = None) -> np.ndarray:
        if not self.is_fitted or self.x_min is None or self.x_scale is None:
            raise RuntimeError("MinMaxScaler must be fitted before transform().")
        x = np.asarray(x, dtype=np.float32)
        if out is None:
            return (self.low + (self.high - self.low) * (x - self.x_min) / self.x_scale).astype(
                np.float32
            )
        np.copyto(out, x, casting="unsafe")
        np.subtract(out, self.x_min, out=out)
        np.divide(out, self.x_scale, out=out)
        np.multiply(out, (self.high - self.low), out=out)
        np.add(out, self.low, out=out)
        return out

    def inverse_transform(self, x: np.ndarray, *, out: np.ndarray | None = None) -> np.ndarray:
        if not self.is_fitted or self.x_min is None or self.x_scale is None:
            raise RuntimeError("MinMaxScaler must be fitted before inverse_transform().")
        x = np.asarray(x, dtype=np.float32)
        if out is None:
            return (((x - self.low) / (self.high - self.low)) * self.x_scale + self.x_min).astype(
                np.float32
            )
        np.copyto(out, x, casting="unsafe")
        np.subtract(out, self.low, out=out)
        np.divide(out, (self.high - self.low), out=out)
        np.multiply(out, self.x_scale, out=out)
        np.add(out, self.x_min, out=out)
        return out


class SelectiveMinMaxScaler(ArrayTransform):
    """Column-wise min-max scaling with pass-through columns."""

    def __init__(
        self,
        *,
        scale_columns: Sequence[bool],
        low: float = 0.0,
        high: float = 1.0,
    ) -> None:
        super().__init__()
        if high <= low:
            raise ValueError("high must be greater than low.")
        mask = np.asarray(scale_columns, dtype=bool)
        if mask.ndim != 1 or mask.size == 0:
            raise ValueError("scale_columns must be a non-empty 1D boolean mask.")
        self.low = float(low)
        self.high = float(high)
        self.scale_columns = mask
        self.x_min: np.ndarray | None = None
        self.x_scale: np.ndarray | None = None

    @classmethod
    def from_passthrough_parameters(
        cls,
        parameters: Sequence[object],
        *,
        passthrough_kinds: Sequence[str] = ("concentration",),
        low: float = 0.0,
        high: float = 1.0,
    ) -> "SelectiveMinMaxScaler":
        normalized_kinds = {
            str(kind).strip().lower().replace("-", "_").replace(" ", "_")
            for kind in passthrough_kinds
        }
        scale_columns = [
            str(getattr(parameter, "kind", "")).strip().lower().replace("-", "_").replace(" ", "_")
            not in normalized_kinds
            for parameter in parameters
        ]
        return cls(scale_columns=scale_columns, low=low, high=high)

    def fit(self, x: np.ndarray) -> "SelectiveMinMaxScaler":
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError("SelectiveMinMaxScaler expects a 2D array.")
        self.input_dimension = int(x.shape[1])
        if self.input_dimension != int(self.scale_columns.size):
            raise ValueError("scale_columns length must match the input column count.")

        x_min = np.min(x, axis=0)
        x_max = np.max(x, axis=0)
        x_scale = x_max - x_min
        x_scale = np.where(x_scale > 0.0, x_scale, 1.0)

        passthrough = ~self.scale_columns
        if np.any(passthrough):
            x_min = x_min.astype(np.float32, copy=False)
            x_scale = x_scale.astype(np.float32, copy=False)
            x_min[passthrough] = self.low
            x_scale[passthrough] = self.high - self.low

        self.x_min = x_min.astype(np.float32, copy=False)
        self.x_scale = x_scale.astype(np.float32, copy=False)
        self.is_fitted = True
        return self

    def transform(self, x: np.ndarray, *, out: np.ndarray | None = None) -> np.ndarray:
        if not self.is_fitted or self.x_min is None or self.x_scale is None:
            raise RuntimeError("SelectiveMinMaxScaler must be fitted before transform().")
        x = np.asarray(x, dtype=np.float32)
        if out is None:
            return (self.low + (self.high - self.low) * (x - self.x_min) / self.x_scale).astype(
                np.float32
            )
        np.copyto(out, x, casting="unsafe")
        np.subtract(out, self.x_min, out=out)
        np.divide(out, self.x_scale, out=out)
        np.multiply(out, (self.high - self.low), out=out)
        np.add(out, self.low, out=out)
        return out

    def inverse_transform(self, x: np.ndarray, *, out: np.ndarray | None = None) -> np.ndarray:
        if not self.is_fitted or self.x_min is None or self.x_scale is None:
            raise RuntimeError("SelectiveMinMaxScaler must be fitted before inverse_transform().")
        x = np.asarray(x, dtype=np.float32)
        if out is None:
            return (((x - self.low) / (self.high - self.low)) * self.x_scale + self.x_min).astype(
                np.float32
            )
        np.copyto(out, x, casting="unsafe")
        np.subtract(out, self.low, out=out)
        np.divide(out, (self.high - self.low), out=out)
        np.multiply(out, self.x_scale, out=out)
        np.add(out, self.x_min, out=out)
        return out


class ParameterBoundMinMaxScaler(ArrayTransform):
    """Column-wise min-max scaling with optional fixed bounds per parameter kind.

    Columns can be handled in three modes:
    - pass-through: copied unchanged by forcing the effective range to ``[low, high]``
    - fixed-bounds: scaled using predeclared source bounds, independent of fit data
    - fitted: scaled from the observed data range during ``fit()``
    """

    def __init__(
        self,
        parameters: Sequence[object],
        *,
        passthrough_kinds: Sequence[str] = ("concentration",),
        fixed_bounds_by_kind: dict[str, tuple[float, float]] | None = None,
        low: float = 0.0,
        high: float = 1.0,
    ) -> None:
        super().__init__()
        if high <= low:
            raise ValueError("high must be greater than low.")
        self.parameters = tuple(parameters)
        self.low = float(low)
        self.high = float(high)
        self.parameter_kinds = tuple(
            str(getattr(parameter, "kind", "")).strip().lower().replace("-", "_").replace(" ", "_")
            for parameter in self.parameters
        )
        self.setup_feature_count = 0
        self.layer_param_size = 0
        self.runtime_prefix_layout = False
        self.passthrough_kinds = {
            str(kind).strip().lower().replace("-", "_").replace(" ", "_")
            for kind in passthrough_kinds
        }
        fixed_bounds_by_kind = fixed_bounds_by_kind or {}
        self.fixed_bounds_by_kind = {
            str(kind).strip().lower().replace("-", "_").replace(" ", "_"): (float(bounds[0]), float(bounds[1]))
            for kind, bounds in fixed_bounds_by_kind.items()
        }
        for kind, (bound_low, bound_high) in self.fixed_bounds_by_kind.items():
            if bound_high <= bound_low:
                raise ValueError(f"Fixed bounds for kind '{kind}' must satisfy high > low.")
        self.scale_columns = np.ones((len(self.parameters),), dtype=bool)
        self.fixed_range_columns = np.zeros((len(self.parameters),), dtype=bool)
        self.x_min: np.ndarray | None = None
        self.x_scale: np.ndarray | None = None
        self._infer_runtime_layout()

    @classmethod
    def from_parameters(
        cls,
        parameters: Sequence[object],
        *,
        passthrough_kinds: Sequence[str] = ("concentration",),
        fixed_bounds_by_kind: dict[str, tuple[float, float]] | None = None,
        low: float = 0.0,
        high: float = 1.0,
    ) -> "ParameterBoundMinMaxScaler":
        return cls(
            parameters,
            passthrough_kinds=passthrough_kinds,
            fixed_bounds_by_kind=fixed_bounds_by_kind,
            low=low,
            high=high,
        )

    def _infer_runtime_layout(self) -> None:
        self.setup_feature_count = sum(
            1 for parameter in self.parameters
            if str(getattr(parameter, "group", "")).strip().lower() == "setup"
        )
        layer_groups: dict[int, list[int]] = {}
        for index, parameter in enumerate(self.parameters):
            group = str(getattr(parameter, "group", "")).strip().lower()
            if group != "layer":
                continue
            layer_index = int(getattr(parameter, "layer_index", 0))
            if layer_index < 1:
                self.runtime_prefix_layout = False
                return
            layer_groups.setdefault(layer_index, []).append(index)

        if not layer_groups:
            self.runtime_prefix_layout = True
            self.layer_param_size = 0
            return

        ordered_layer_indices = sorted(layer_groups.keys())
        if ordered_layer_indices != list(range(1, ordered_layer_indices[-1] + 1)):
            self.runtime_prefix_layout = False
            return

        template = layer_groups[ordered_layer_indices[0]]
        self.layer_param_size = len(template)
        if self.layer_param_size <= 0:
            self.runtime_prefix_layout = False
            return

        offset = self.setup_feature_count
        for layer_index in ordered_layer_indices:
            current = layer_groups[layer_index]
            expected = list(range(offset, offset + self.layer_param_size))
            if current != expected or len(current) != self.layer_param_size:
                self.runtime_prefix_layout = False
                return
            offset += self.layer_param_size
        self.runtime_prefix_layout = True

    def _runtime_layer_count(self, feature_count: int) -> int:
        if feature_count < self.setup_feature_count:
            raise ValueError(
                f"Expected at least {self.setup_feature_count} setup features, got {feature_count}."
            )
        if self.layer_param_size <= 0:
            return 0
        layer_feature_count = feature_count - self.setup_feature_count
        if layer_feature_count % self.layer_param_size != 0:
            raise ValueError(
                "Variable-width inputs must use the layout "
                f"setup + N * layer_features, where layer_features={self.layer_param_size}. "
                f"Got {feature_count} features."
            )
        return layer_feature_count // self.layer_param_size

    def fit(self, x: np.ndarray) -> "ParameterBoundMinMaxScaler":
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError("ParameterBoundMinMaxScaler expects a 2D array.")
        self.input_dimension = int(x.shape[1])
        if self.input_dimension != len(self.parameters):
            raise ValueError("Parameter count must match the input column count.")

        x_min = np.min(x, axis=0).astype(np.float32, copy=False)
        x_max = np.max(x, axis=0).astype(np.float32, copy=False)
        x_scale = np.where((x_max - x_min) > 0.0, x_max - x_min, 1.0).astype(np.float32, copy=False)

        scale_columns = np.ones((self.input_dimension,), dtype=bool)
        fixed_range_columns = np.zeros((self.input_dimension,), dtype=bool)

        for index, kind in enumerate(self.parameter_kinds):
            if kind in self.passthrough_kinds:
                scale_columns[index] = False
                x_min[index] = self.low
                x_scale[index] = self.high - self.low
                continue
            bounds = self.fixed_bounds_by_kind.get(kind)
            if bounds is None:
                continue
            bound_low, bound_high = bounds
            fixed_range_columns[index] = True
            x_min[index] = np.float32(bound_low)
            x_scale[index] = np.float32(bound_high - bound_low)

        self.scale_columns = scale_columns
        self.fixed_range_columns = fixed_range_columns
        self.x_min = x_min
        self.x_scale = x_scale
        self.is_fitted = True
        return self

    def transform(self, x: np.ndarray, *, out: np.ndarray | None = None) -> np.ndarray:
        if not self.is_fitted or self.x_min is None or self.x_scale is None:
            raise RuntimeError("ParameterBoundMinMaxScaler must be fitted before transform().")
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError("ParameterBoundMinMaxScaler expects a 2D array.")
        if x.shape[1] == self.input_dimension:
            if out is None:
                return (self.low + (self.high - self.low) * (x - self.x_min) / self.x_scale).astype(
                    np.float32
                )
            np.copyto(out, x, casting="unsafe")
            np.subtract(out, self.x_min, out=out)
            np.divide(out, self.x_scale, out=out)
            np.multiply(out, (self.high - self.low), out=out)
            np.add(out, self.low, out=out)
            return out

        if not self.runtime_prefix_layout:
            raise ValueError(
                "Variable-width parameter scaling requires setup features followed by contiguous "
                "per-layer feature blocks in parameter order."
            )

        layer_count = self._runtime_layer_count(int(x.shape[1]))
        if out is None:
            result = np.array(x, dtype=np.float32, copy=True)
        else:
            np.copyto(out, x, casting="unsafe")
            result = out

        if self.setup_feature_count > 0:
            result[:, : self.setup_feature_count] = (
                self.low
                + (self.high - self.low)
                * (result[:, : self.setup_feature_count] - self.x_min[: self.setup_feature_count])
                / self.x_scale[: self.setup_feature_count]
            )

        if self.layer_param_size > 0 and layer_count > 0:
            layer_values = result[:, self.setup_feature_count :].reshape(
                result.shape[0],
                layer_count,
                self.layer_param_size,
            )
            template_min = self.x_min[
                self.setup_feature_count : self.setup_feature_count + self.layer_param_size
            ].reshape(1, 1, -1)
            template_scale = self.x_scale[
                self.setup_feature_count : self.setup_feature_count + self.layer_param_size
            ].reshape(1, 1, -1)
            layer_values -= template_min
            layer_values /= template_scale
            layer_values *= (self.high - self.low)
            layer_values += self.low
        return result

    def inverse_transform(self, x: np.ndarray, *, out: np.ndarray | None = None) -> np.ndarray:
        if not self.is_fitted or self.x_min is None or self.x_scale is None:
            raise RuntimeError("ParameterBoundMinMaxScaler must be fitted before inverse_transform().")
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError("ParameterBoundMinMaxScaler expects a 2D array.")
        if x.shape[1] == self.input_dimension:
            if out is None:
                return (((x - self.low) / (self.high - self.low)) * self.x_scale + self.x_min).astype(
                    np.float32
                )
            np.copyto(out, x, casting="unsafe")
            np.subtract(out, self.low, out=out)
            np.divide(out, (self.high - self.low), out=out)
            np.multiply(out, self.x_scale, out=out)
            np.add(out, self.x_min, out=out)
            return out

        if not self.runtime_prefix_layout:
            raise ValueError(
                "Variable-width parameter scaling requires setup features followed by contiguous "
                "per-layer feature blocks in parameter order."
            )

        layer_count = self._runtime_layer_count(int(x.shape[1]))
        if out is None:
            result = np.array(x, dtype=np.float32, copy=True)
        else:
            np.copyto(out, x, casting="unsafe")
            result = out

        if self.setup_feature_count > 0:
            result[:, : self.setup_feature_count] = (
                ((result[:, : self.setup_feature_count] - self.low) / (self.high - self.low))
                * self.x_scale[: self.setup_feature_count]
                + self.x_min[: self.setup_feature_count]
            )

        if self.layer_param_size > 0 and layer_count > 0:
            layer_values = result[:, self.setup_feature_count :].reshape(
                result.shape[0],
                layer_count,
                self.layer_param_size,
            )
            template_min = self.x_min[
                self.setup_feature_count : self.setup_feature_count + self.layer_param_size
            ].reshape(1, 1, -1)
            template_scale = self.x_scale[
                self.setup_feature_count : self.setup_feature_count + self.layer_param_size
            ].reshape(1, 1, -1)
            layer_values -= self.low
            layer_values /= (self.high - self.low)
            layer_values *= template_scale
            layer_values += template_min
        return result


class LayerwiseConcentrationNormalizer(ArrayTransform):
    """Normalize concentration features to sum to one within each layer block.

    The transform is intended for model inputs whose columns follow the same
    layer-wise layout as ``build_lrn_model_schema()``. It supports both the
    fixed-width training matrix and eval-time variable-width inputs that follow
    the compact ``setup + N * layer_features`` layout.

    The operation is not invertible in general. ``inverse_transform()`` is
    therefore implemented as a pass-through copy.
    """

    def __init__(
        self,
        *,
        setup_feature_count: int,
        layer_param_size: int,
        layer_feature_indices: Sequence[Sequence[int]],
        concentration_indices: Sequence[Sequence[int]],
        runtime_prefix_layout: bool = False,
    ) -> None:
        super().__init__()
        self.setup_feature_count = int(setup_feature_count)
        self.layer_param_size = int(layer_param_size)
        self.layer_feature_indices = tuple(tuple(int(index) for index in group) for group in layer_feature_indices)
        self.concentration_indices = tuple(
            tuple(int(index) for index in group)
            for group in concentration_indices
        )
        self.runtime_prefix_layout = bool(runtime_prefix_layout)
        if len(self.layer_feature_indices) != len(self.concentration_indices):
            raise ValueError("layer_feature_indices and concentration_indices must have matching lengths.")
        if self.layer_param_size < 0:
            raise ValueError("layer_param_size must be >= 0.")
        if self.setup_feature_count < 0:
            raise ValueError("setup_feature_count must be >= 0.")

    @classmethod
    def from_schema(cls, schema: object) -> "LayerwiseConcentrationNormalizer":
        from ibamlkit.models.forward.lrn import LRNModel

        layout = LRNModel._infer_input_layout(schema)
        runtime_prefix_layout = LRNModel._has_prefix_runtime_layout(layout)
        return cls(
            setup_feature_count=layout.setup_param_size,
            layer_param_size=layout.layer_param_size,
            layer_feature_indices=layout.layer_feature_indices,
            concentration_indices=layout.concentration_feature_offsets,
            runtime_prefix_layout=runtime_prefix_layout,
        )

    def fit(self, x: np.ndarray) -> "LayerwiseConcentrationNormalizer":
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError("LayerwiseConcentrationNormalizer expects a 2D array.")
        self.input_dimension = int(x.shape[1])
        self.is_fitted = True
        return self

    @staticmethod
    def _normalize_block(block: np.ndarray, concentration_offsets: Sequence[int]) -> np.ndarray:
        if not concentration_offsets:
            return block
        concentrations = np.clip(block[:, concentration_offsets], a_min=0.0, a_max=None)
        sums = concentrations.sum(axis=1, keepdims=True)
        safe_sums = np.where(sums > 0.0, sums, 1.0)
        block[:, concentration_offsets] = concentrations / safe_sums
        return block

    def _runtime_layer_count(self, feature_count: int) -> int:
        if feature_count < self.setup_feature_count:
            raise ValueError(
                f"Expected at least {self.setup_feature_count} setup features, got {feature_count}."
            )
        if self.layer_param_size <= 0:
            return 0
        layer_feature_count = feature_count - self.setup_feature_count
        if layer_feature_count % self.layer_param_size != 0:
            raise ValueError(
                "Variable-length inputs must use the layout "
                f"setup + N * layer_features, where layer_features={self.layer_param_size}. "
                f"Got {feature_count} features."
            )
        return layer_feature_count // self.layer_param_size

    def transform(self, x: np.ndarray, *, out: np.ndarray | None = None) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("LayerwiseConcentrationNormalizer must be fitted before transform().")
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError("LayerwiseConcentrationNormalizer expects a 2D array.")

        if out is None:
            result = np.array(x, dtype=np.float32, copy=True)
        else:
            np.copyto(out, x, casting="unsafe")
            result = out

        if not self.concentration_indices or self.layer_param_size <= 0:
            return result

        if self.input_dimension is not None and result.shape[1] == self.input_dimension:
            for layer_indices, concentration_offsets in zip(
                self.layer_feature_indices,
                self.concentration_indices,
            ):
                if not concentration_offsets:
                    continue
                block = np.array(result[:, layer_indices], dtype=np.float32, copy=True)
                block = self._normalize_block(block, concentration_offsets)
                result[:, layer_indices] = block
            return result

        if not self.runtime_prefix_layout:
            raise ValueError(
                "Variable-length concentration normalization requires setup features followed by "
                "contiguous per-layer feature blocks in schema order."
            )

        concentration_offsets = self.concentration_indices[0]
        if not concentration_offsets:
            return result
        for layer_index in range(self._runtime_layer_count(int(result.shape[1]))):
            start = self.setup_feature_count + layer_index * self.layer_param_size
            stop = start + self.layer_param_size
            result[:, start:stop] = self._normalize_block(
                np.array(result[:, start:stop], dtype=np.float32, copy=True),
                concentration_offsets,
            )
        return result

    def inverse_transform(self, x: np.ndarray, *, out: np.ndarray | None = None) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if out is None:
            return np.array(x, dtype=np.float32, copy=True)
        np.copyto(out, x, casting="unsafe")
        return out


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

    def transform(self, x: np.ndarray, *, out: np.ndarray | None = None) -> np.ndarray:
        if not self.is_fitted or self.x_mean is None or self.x_std is None:
            raise RuntimeError("StandardScaler must be fitted before transform().")
        x = np.asarray(x, dtype=np.float32)
        if out is None:
            return ((x - self.x_mean) / self.x_std).astype(np.float32)
        np.copyto(out, x, casting="unsafe")
        np.subtract(out, self.x_mean, out=out)
        np.divide(out, self.x_std, out=out)
        return out

    def inverse_transform(self, x: np.ndarray, *, out: np.ndarray | None = None) -> np.ndarray:
        if not self.is_fitted or self.x_mean is None or self.x_std is None:
            raise RuntimeError("StandardScaler must be fitted before inverse_transform().")
        x = np.asarray(x, dtype=np.float32)
        if out is None:
            return (x * self.x_std + self.x_mean).astype(np.float32)
        np.copyto(out, x, casting="unsafe")
        np.multiply(out, self.x_std, out=out)
        np.add(out, self.x_mean, out=out)
        return out


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

    def transform(self, x: np.ndarray, *, out: np.ndarray | None = None) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("TransformPipeline must be fitted before transform().")
        x_work = np.asarray(x, dtype=np.float32)
        for transform in self.transforms:
            x_work = transform.transform(x_work, out=out if transform is self.transforms[-1] else None)
        return x_work.astype(np.float32)

    def inverse_transform(self, x: np.ndarray, *, out: np.ndarray | None = None) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("TransformPipeline must be fitted before inverse_transform().")
        x_work = np.asarray(x, dtype=np.float32)
        for transform in reversed(self.transforms):
            x_work = transform.inverse_transform(x_work, out=out if transform is self.transforms[0] else None)
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
