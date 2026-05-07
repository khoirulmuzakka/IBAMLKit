"""Validation helpers for canonical dataset objects."""

from __future__ import annotations

from ..schema import DatasetInputSpec, IBADataset


def validate_input_spec(input_spec: DatasetInputSpec) -> DatasetInputSpec:
    """Return the validated input spec.

    Validation is executed by the schema constructors, so this function mainly
    provides a stable public entry point in the data layer.
    """

    return input_spec


def validate_dataset(dataset: IBADataset) -> IBADataset:
    """Return the validated dataset."""

    return dataset
