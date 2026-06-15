"""Export trained forward models as an ONNX runtime package."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping
import shutil
import tempfile
import zipfile

import numpy as np
import torch

from ibamlkit.schema import DatasetInputSpec, ForwardModelSchema, IBADataset
from ibamlkit.training.preprocessing import (
    ArrayTransform,
    ConstantFactorTransform,
    IdentityTransform,
    LayerwiseConcentrationNormalizer,
    MinMaxScaler,
    ParameterBoundMinMaxScaler,
    SelectiveMinMaxScaler,
    StandardScaler,
    TransformPipeline,
)


@dataclass(frozen=True)
class ModelPackageArtifacts:
    """Auxiliary state needed to run a trained model outside Python.

    The ONNX graph covers the neural network. This container stores the
    dataset contract and fitted preprocessing state that the external runtime
    must reproduce exactly.
    """

    input_spec: DatasetInputSpec
    input_scaler: ArrayTransform | None = None
    output_transform: ArrayTransform | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    example_input: np.ndarray | torch.Tensor | None = None


def export_as_package(
    model: torch.nn.Module,
    package_data: ModelPackageArtifacts | DatasetInputSpec | IBADataset | Mapping[str, Any],
    filepath: str | Path,
    *,
    opset_version: int = 17,
    zip_archive: bool = False,
) -> Path:
    """Export a trained model as ``model.onnx`` plus a JSON manifest.

    Parameters
    ----------
    model:
        Trained forward model to export.
    package_data:
        Dataset contract and preprocessing state. This must include at least an
        ``input_spec``. A plain :class:`IBADataset` is also accepted and its
        ``input_spec`` is extracted automatically.
    filepath:
        Output directory, or a path ending in ``.onnx``. In directory mode the
        function creates ``model.onnx`` and ``package.json`` inside that folder.
    opset_version:
        ONNX opset to target.
    zip_archive:
        If ``True``, write a ``.zip`` archive containing the package contents.
        When ``filepath`` ends in ``.zip`` this flag is implied.

    Returns
    -------
    Path
        The output directory containing the exported package.
    """

    artifacts = _coerce_artifacts(package_data)
    package_dir, onnx_path, manifest_path, archive_path = _resolve_output_paths(
        filepath,
        zip_archive=zip_archive,
    )
    package_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_package_manifest(
        model=model,
        artifacts=artifacts,
        onnx_filename=onnx_path.name,
        opset_version=opset_version,
    )

    export_model = model
    if hasattr(export_model, "module"):
        export_model = export_model.module  # type: ignore[assignment]

    original_device = _model_device(export_model)
    was_training = export_model.training if isinstance(export_model, torch.nn.Module) else False
    try:
        export_model = export_model.to("cpu")
        export_model.eval()
        example_input = _resolve_example_input(export_model, artifacts)
        example_input = example_input.to(dtype=torch.float32, device="cpu")
        with torch.no_grad():
            try:
                torch.onnx.export(
                    export_model,
                    example_input,
                    str(onnx_path),
                    input_names=["inputs"],
                    output_names=["outputs"],
                    dynamic_axes={"inputs": {0: "batch"}, "outputs": {0: "batch"}},
                    opset_version=opset_version,
                    do_constant_folding=True,
                    export_params=True,
                )
            except Exception as exc:  # pragma: no cover - dependency-specific error path
                if "Module onnx is not installed" in str(exc):
                    raise RuntimeError(
                        "ONNX export requires the `onnx` package in the export environment."
                    ) from exc
                raise
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        if archive_path is not None:
            with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.write(onnx_path, arcname=onnx_path.name)
                archive.write(manifest_path, arcname=manifest_path.name)
            return archive_path
        return package_dir
    finally:
        if original_device is not None:
            export_model.to(original_device)
        if was_training:
            export_model.train()
        if archive_path is not None:
            shutil.rmtree(package_dir, ignore_errors=True)


def build_package_manifest(
    *,
    model: torch.nn.Module,
    artifacts: ModelPackageArtifacts,
    onnx_filename: str = "model.onnx",
    opset_version: int = 17,
) -> dict[str, Any]:
    """Build the JSON manifest for a packaged ONNX model."""

    schema = getattr(model, "schema", None)
    if not isinstance(schema, ForwardModelSchema):
        raise TypeError("export_as_package expects a forward model with a ForwardModelSchema.")

    input_spec_dict = _dataset_input_spec_to_dict(artifacts.input_spec)
    model_schema_dict = _model_schema_to_dict(schema)
    output_dimension = int(schema.outputs.feature_dimension)
    if output_dimension <= 0 and schema.outputs.spectra_lengths:
        output_dimension = int(sum(schema.outputs.spectra_lengths.values()))
    manifest = {
        "format": "ibamlkit.onnx-package",
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "class_name": model.__class__.__name__,
            "onnx_file": onnx_filename,
            "opset_version": int(opset_version),
            "input_dimension": int(schema.inputs.dimension),
            "output_dimension": output_dimension,
            "schema": model_schema_dict,
        },
        "dataset": input_spec_dict,
        "preprocessing": {
            "input_scaler": _transform_to_dict(artifacts.input_scaler),
            "output_transform": _transform_to_dict(artifacts.output_transform),
        },
        "metadata": _jsonable(dict(artifacts.metadata)),
    }
    return manifest


def _coerce_artifacts(
    package_data: ModelPackageArtifacts | DatasetInputSpec | IBADataset | Mapping[str, Any],
) -> ModelPackageArtifacts:
    if isinstance(package_data, ModelPackageArtifacts):
        return package_data
    if isinstance(package_data, IBADataset):
        return ModelPackageArtifacts(input_spec=package_data.input_spec)
    if isinstance(package_data, DatasetInputSpec):
        return ModelPackageArtifacts(input_spec=package_data)
    if isinstance(package_data, Mapping):
        input_spec = package_data.get("input_spec") or package_data.get("dataset_input_spec")
        if not isinstance(input_spec, DatasetInputSpec):
            raise TypeError(
                "package_data mapping must contain an 'input_spec' or 'dataset_input_spec' entry."
            )
        input_scaler = package_data.get("input_scaler")
        output_transform = package_data.get("output_transform")
        metadata = package_data.get("metadata", {})
        example_input = package_data.get("example_input")
        return ModelPackageArtifacts(
            input_spec=input_spec,
            input_scaler=input_scaler if isinstance(input_scaler, ArrayTransform) else None,
            output_transform=(
                output_transform if isinstance(output_transform, ArrayTransform) else None
            ),
            metadata=metadata if isinstance(metadata, Mapping) else dict(metadata),
            example_input=example_input,
        )
    raise TypeError(
        "package_data must be a ModelPackageArtifacts, DatasetInputSpec, IBADataset, or mapping."
    )


def _resolve_output_paths(filepath: str | Path, *, zip_archive: bool = False) -> tuple[Path, Path, Path, Path | None]:
    path = Path(filepath)
    archive_path: Path | None = None
    if path.suffix.lower() == ".zip" or zip_archive:
        archive_path = path if path.suffix.lower() == ".zip" else Path(str(path) + ".zip")
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        package_dir = Path(
            tempfile.mkdtemp(prefix=f"{archive_path.stem}_", dir=str(archive_path.parent or Path.cwd()))
        )
        onnx_path = package_dir / "model.onnx"
        manifest_path = package_dir / "package.json"
        return package_dir, onnx_path, manifest_path, archive_path
    if path.suffix.lower() == ".onnx":
        package_dir = path.parent
        onnx_path = path
        manifest_path = path.with_suffix(".json")
    else:
        package_dir = path
        onnx_path = package_dir / "model.onnx"
        manifest_path = package_dir / "package.json"
    return package_dir, onnx_path, manifest_path, None


def _resolve_example_input(
    model: torch.nn.Module,
    artifacts: ModelPackageArtifacts,
) -> torch.Tensor:
    if artifacts.example_input is not None:
        example = torch.as_tensor(artifacts.example_input, dtype=torch.float32)
        if example.ndim != 2:
            raise ValueError("example_input must be a 2D array or tensor.")
        return example
    if hasattr(model, "example_input"):
        example = model.example_input(batch_size=1)  # type: ignore[call-arg]
        if not isinstance(example, torch.Tensor):
            example = torch.as_tensor(example, dtype=torch.float32)
        return example
    raise TypeError("Model must provide example_input() or package_data must include example_input.")


def _model_device(model: torch.nn.Module) -> torch.device | None:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return None


def _dataset_input_spec_to_dict(input_spec: DatasetInputSpec) -> dict[str, Any]:
    return _jsonable(asdict(input_spec))


def _model_schema_to_dict(schema: ForwardModelSchema) -> dict[str, Any]:
    return _jsonable(asdict(schema))


def _transform_to_dict(transform: ArrayTransform | None) -> dict[str, Any] | None:
    if transform is None:
        return None
    if isinstance(transform, IdentityTransform):
        return {
            "type": "IdentityTransform",
            "is_fitted": bool(transform.is_fitted),
            "input_dimension": transform.input_dimension,
        }
    if isinstance(transform, ConstantFactorTransform):
        return {
            "type": "ConstantFactorTransform",
            "is_fitted": bool(transform.is_fitted),
            "input_dimension": transform.input_dimension,
            "factor": float(transform.factor),
        }
    if isinstance(transform, MinMaxScaler):
        return {
            "type": "MinMaxScaler",
            "is_fitted": bool(transform.is_fitted),
            "input_dimension": transform.input_dimension,
            "low": float(transform.low),
            "high": float(transform.high),
            "x_min": None if transform.x_min is None else transform.x_min.astype(np.float32).tolist(),
            "x_scale": None
            if transform.x_scale is None
            else transform.x_scale.astype(np.float32).tolist(),
        }
    if isinstance(transform, SelectiveMinMaxScaler):
        return {
            "type": "SelectiveMinMaxScaler",
            "is_fitted": bool(transform.is_fitted),
            "input_dimension": transform.input_dimension,
            "low": float(transform.low),
            "high": float(transform.high),
            "scale_columns": transform.scale_columns.astype(bool).tolist(),
            "x_min": None if transform.x_min is None else transform.x_min.astype(np.float32).tolist(),
            "x_scale": None
            if transform.x_scale is None
            else transform.x_scale.astype(np.float32).tolist(),
        }
    if isinstance(transform, ParameterBoundMinMaxScaler):
        return {
            "type": "ParameterBoundMinMaxScaler",
            "is_fitted": bool(transform.is_fitted),
            "input_dimension": transform.input_dimension,
            "low": float(transform.low),
            "high": float(transform.high),
            "parameter_kinds": list(transform.parameter_kinds),
            "setup_feature_count": int(transform.setup_feature_count),
            "layer_param_size": int(transform.layer_param_size),
            "runtime_prefix_layout": bool(transform.runtime_prefix_layout),
            "passthrough_kinds": sorted(transform.passthrough_kinds),
            "fixed_bounds_by_kind": {
                kind: [float(bounds[0]), float(bounds[1])]
                for kind, bounds in transform.fixed_bounds_by_kind.items()
            },
            "scale_columns": transform.scale_columns.astype(bool).tolist(),
            "fixed_range_columns": transform.fixed_range_columns.astype(bool).tolist(),
            "x_min": None if transform.x_min is None else transform.x_min.astype(np.float32).tolist(),
            "x_scale": None
            if transform.x_scale is None
            else transform.x_scale.astype(np.float32).tolist(),
        }
    if isinstance(transform, LayerwiseConcentrationNormalizer):
        return {
            "type": "LayerwiseConcentrationNormalizer",
            "is_fitted": bool(transform.is_fitted),
            "input_dimension": transform.input_dimension,
            "setup_feature_count": int(transform.setup_feature_count),
            "layer_param_size": int(transform.layer_param_size),
            "layer_feature_indices": [list(group) for group in transform.layer_feature_indices],
            "concentration_indices": [list(group) for group in transform.concentration_indices],
            "runtime_prefix_layout": bool(transform.runtime_prefix_layout),
        }
    if isinstance(transform, StandardScaler):
        return {
            "type": "StandardScaler",
            "is_fitted": bool(transform.is_fitted),
            "input_dimension": transform.input_dimension,
            "x_mean": None
            if transform.x_mean is None
            else transform.x_mean.astype(np.float32).tolist(),
            "x_std": None if transform.x_std is None else transform.x_std.astype(np.float32).tolist(),
        }
    if isinstance(transform, TransformPipeline):
        return {
            "type": "TransformPipeline",
            "is_fitted": bool(transform.is_fitted),
            "input_dimension": transform.input_dimension,
            "transforms": [_transform_to_dict(item) for item in transform.transforms],
        }
    raise TypeError(f"Unsupported transform type: {transform.__class__.__name__}")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return [_jsonable(item) for item in sorted(value, key=lambda item: str(item))]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return value
