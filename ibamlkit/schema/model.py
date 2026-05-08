"""Schema objects describing the public model contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
import json

from .dataset import ParameterSpec


@dataclass(frozen=True)
class ModelTaskSpec:
    """High-level description of what a model does."""

    task_kind: str
    method_names: Sequence[str]
    package_version: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.task_kind not in {"surrogate", "inverse"}:
            raise ValueError("task_kind must be either 'surrogate' or 'inverse'.")
        if not self.method_names:
            raise ValueError("At least one method name must be declared.")
        if len(self.method_names) != len(set(self.method_names)):
            raise ValueError("Method names must be unique.")
        json.dumps(dict(self.metadata))


@dataclass(frozen=True)
class TensorFeatureSpec:
    """One ordered model-side feature derived from a physical parameter."""

    name: str
    source_parameter: str
    role: str
    transform: str = "identity"
    group: str = ""
    layer_index: int = -1
    unit: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Feature name must not be empty.")
        if not self.source_parameter:
            raise ValueError("source_parameter must not be empty.")
        if self.role not in {"input", "output"}:
            raise ValueError("role must be either 'input' or 'output'.")
        if self.group and self.group not in {"setup", "layer", "spectrum", "derived"}:
            raise ValueError(
                "group must be empty or one of 'setup', 'layer', 'spectrum', 'derived'."
            )
        if self.layer_index == 0 or self.layer_index < -1:
            raise ValueError("layer_index must be -1 or a 1-based positive integer.")
        json.dumps(dict(self.metadata))


@dataclass(frozen=True)
class ModelInputSpec:
    """Ordered input-tensor contract for a model."""

    features: Sequence[TensorFeatureSpec]
    layout: str = "flat"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.layout not in {"flat", "grouped_by_layer"}:
            raise ValueError("layout must be either 'flat' or 'grouped_by_layer'.")
        if not self.features:
            raise ValueError("At least one input feature must be defined.")
        if any(feature.role != "input" for feature in self.features):
            raise ValueError("All ModelInputSpec features must have role='input'.")
        feature_names = [feature.name for feature in self.features]
        if len(feature_names) != len(set(feature_names)):
            raise ValueError("Input feature names must be unique.")
        json.dumps(dict(self.metadata))

    @property
    def dimension(self) -> int:
        return len(self.features)


@dataclass(frozen=True)
class ModelOutputSpec:
    """Ordered output-tensor contract for a model."""

    features: Sequence[TensorFeatureSpec] = field(default_factory=tuple)
    spectra_names: Sequence[str] = field(default_factory=tuple)
    spectra_lengths: Mapping[str, int] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.features and not self.spectra_names:
            raise ValueError("Define output features, spectra_names, or both.")
        if any(feature.role != "output" for feature in self.features):
            raise ValueError("All ModelOutputSpec features must have role='output'.")

        feature_names = [feature.name for feature in self.features]
        if len(feature_names) != len(set(feature_names)):
            raise ValueError("Output feature names must be unique.")

        if self.spectra_names:
            if len(self.spectra_names) != len(set(self.spectra_names)):
                raise ValueError("spectra_names must be unique.")
            if set(self.spectra_lengths.keys()) != set(self.spectra_names):
                raise ValueError(
                    "spectra_lengths keys must exactly match spectra_names when spectra are declared."
                )
            if any(length <= 0 for length in self.spectra_lengths.values()):
                raise ValueError("Every declared spectrum length must be positive.")
        elif self.spectra_lengths:
            raise ValueError("spectra_lengths requires spectra_names to be declared.")

        json.dumps(dict(self.metadata))

    @property
    def feature_dimension(self) -> int:
        return len(self.features)


@dataclass(frozen=True)
class ModelSchema:
    """Top-level public description of a model package contract."""

    name: str
    task: ModelTaskSpec
    inputs: ModelInputSpec
    outputs: ModelOutputSpec
    parameters: Sequence[ParameterSpec] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Model name must not be empty.")

        parameter_names = [parameter.name for parameter in self.parameters]
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError("Model parameter names must be unique.")

        known_parameters = set(parameter_names)
        if known_parameters:
            for feature in tuple(self.inputs.features) + tuple(self.outputs.features):
                if feature.source_parameter not in known_parameters:
                    raise ValueError(
                        f"Feature '{feature.name}' references unknown parameter "
                        f"'{feature.source_parameter}'."
                    )

        if self.outputs.spectra_names:
            missing = set(self.outputs.spectra_names) - set(self.task.method_names)
            if missing:
                raise ValueError(
                    "Every declared output spectrum must map to a method in task.method_names."
                )

        json.dumps(dict(self.metadata))
