"""Helpers for building inverse-model schemas from canonical datasets."""

from __future__ import annotations

from typing import Sequence

from ibamlkit.schema import (
    IBADataset,
    InverseModelSchema,
    ModelInputSpec,
    ModelOutputSpec,
    ModelTaskSpec,
    TensorFeatureSpec,
)


def build_inverse_model_schema(
    dataset: IBADataset,
    *,
    method_name: str,
    model_name: str = "inverse_mlp",
    target_parameter_names: Sequence[str] | None = None,
) -> InverseModelSchema:
    """Build a default inverse schema from one canonical dataset."""

    if method_name not in dataset.spectra:
        raise KeyError(f"Method {method_name!r} not found in dataset spectra.")

    spectrum_width = int(dataset.spectra[method_name].shape[1])
    input_features = [
        TensorFeatureSpec(
            name=f"{method_name}_channel_{channel_index}",
            source_parameter=f"{method_name}_channel_{channel_index}",
            role="input",
            group="spectrum",
            metadata={"method": method_name, "channel_index": channel_index},
        )
        for channel_index in range(spectrum_width)
    ]

    open_parameters = list(dataset.input_spec.open_parameters)
    if target_parameter_names is None:
        selected_parameters = open_parameters
    else:
        parameter_map = {parameter.name: parameter for parameter in open_parameters}
        selected_parameters = []
        for name in target_parameter_names:
            try:
                selected_parameters.append(parameter_map[name])
            except KeyError as exc:
                raise KeyError(f"Unknown open parameter {name!r}.") from exc

    output_features = [
        TensorFeatureSpec(
            name=parameter.name,
            source_parameter=parameter.name,
            role="output",
            group=parameter.group,
            layer_index=parameter.layer_index,
            unit=parameter.unit,
            metadata={
                "kind": parameter.kind,
                "method": parameter.method,
                "element": parameter.element,
                "isotope": parameter.isotope,
            },
        )
        for parameter in selected_parameters
    ]

    return InverseModelSchema(
        name=model_name,
        task=ModelTaskSpec(task_kind="inverse", method_names=[method_name]),
        inputs=ModelInputSpec(features=input_features, layout="flat"),
        outputs=ModelOutputSpec(features=output_features),
        parameters=dataset.input_spec.parameters,
        metadata={"source": "IBADataset", "method_name": method_name},
    )
