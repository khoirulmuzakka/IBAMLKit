from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest
import torch
from torch import nn

from ibamlkit.schema import (
    DatasetInputSpec,
    ForwardModelSchema,
    LayerSpeciesSpec,
    MethodSpec,
    ModelInputSpec,
    ModelOutputSpec,
    ModelTaskSpec,
    ParameterSpec,
    TensorFeatureSpec,
)
from ibamlkit.training import ConstantFactorTransform, MinMaxScaler, ModelPackageArtifacts, export_as_package
from ibamlkit.models.forward.base import ForwardModelBase


class _DummyForwardModel(ForwardModelBase):
    def __init__(self) -> None:
        schema = ForwardModelSchema(
            name="dummy",
            task=ModelTaskSpec(task_kind="surrogate", method_names=["m"]),
            inputs=ModelInputSpec(
                features=[
                    TensorFeatureSpec(
                        name="x0",
                        source_parameter="x0",
                        role="input",
                    ),
                    TensorFeatureSpec(
                        name="x1",
                        source_parameter="x1",
                        role="input",
                    ),
                ]
            ),
            outputs=ModelOutputSpec(
                features=[
                    TensorFeatureSpec(
                        name="y0",
                        source_parameter="y0",
                        role="output",
                    )
                ]
            ),
            parameters=[],
        )
        super().__init__(schema)
        self.linear = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[1.0, 2.0]]))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.validate_input_shape(inputs)
        return self.linear(inputs)


def test_export_as_package_writes_onnx_and_manifest(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = _DummyForwardModel()

    input_spec = DatasetInputSpec(
        methods=[MethodSpec(name="m")],
        layer_species=[LayerSpeciesSpec(layer_index=1, element="Si")],
        parameters=[
            ParameterSpec(name="x0", group="setup", kind="scalar", is_open=True),
            ParameterSpec(name="x1", group="setup", kind="scalar", is_open=False, fixed_value=3.5),
        ],
        generation_info={"source": "unit-test"},
    )

    input_scaler = MinMaxScaler(low=-1.0, high=1.0).fit(np.array([[0.0], [2.0]], dtype=np.float32))
    output_transform = ConstantFactorTransform(2.5).fit(np.zeros((1, 1), dtype=np.float32))

    def _fake_onnx_export(*args, **kwargs) -> None:
        output_path = args[2]
        with open(output_path, "wb") as handle:
            handle.write(b"onnx-placeholder")

    monkeypatch.setattr(torch.onnx, "export", _fake_onnx_export)

    package_dir = export_as_package(
        model,
        ModelPackageArtifacts(
            input_spec=input_spec,
            input_scaler=input_scaler,
            output_transform=output_transform,
            metadata={"experiment": "demo"},
        ),
        tmp_path,
    )

    onnx_path = package_dir / "model.onnx"
    manifest_path = package_dir / "package.json"
    assert onnx_path.exists()
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["format"] == "ibamlkit.onnx-package"
    assert manifest["dataset"]["generation_info"]["source"] == "unit-test"
    assert manifest["dataset"]["parameters"][0]["name"] == "x0"
    assert manifest["preprocessing"]["input_scaler"]["type"] == "MinMaxScaler"
    assert manifest["preprocessing"]["output_transform"]["type"] == "ConstantFactorTransform"
    assert manifest["metadata"]["experiment"] == "demo"


def test_export_as_package_writes_zip_archive(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = _DummyForwardModel()
    input_spec = DatasetInputSpec(
        methods=[MethodSpec(name="m")],
        layer_species=[LayerSpeciesSpec(layer_index=1, element="Si")],
        parameters=[
            ParameterSpec(name="x0", group="setup", kind="scalar", is_open=True),
            ParameterSpec(name="x1", group="setup", kind="scalar", is_open=False, fixed_value=3.5),
        ],
        generation_info={"source": "unit-test"},
    )

    def _fake_onnx_export(*args, **kwargs) -> None:
        output_path = args[2]
        with open(output_path, "wb") as handle:
            handle.write(b"onnx-placeholder")

    monkeypatch.setattr(torch.onnx, "export", _fake_onnx_export)

    archive_path = export_as_package(
        model,
        input_spec,
        tmp_path / "bundle.zip",
    )

    assert archive_path.suffix == ".zip"
    assert archive_path.exists()
    with zipfile.ZipFile(archive_path, "r") as archive:
        assert sorted(archive.namelist()) == ["model.onnx", "package.json"]
        manifest = json.loads(archive.read("package.json").decode("utf-8"))
    assert manifest["dataset"]["generation_info"]["source"] == "unit-test"
