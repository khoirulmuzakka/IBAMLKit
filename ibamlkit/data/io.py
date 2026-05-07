"""High-level dataset I/O helpers."""

from __future__ import annotations

from ..schema import IBADataset
from .readers import read_hdf5_dataset
from .writers import write_hdf5_dataset


def save_dataset(path: str, dataset: IBADataset) -> None:
    """Write a canonical dataset to HDF5."""

    write_hdf5_dataset(path, dataset)


def load_dataset(path: str) -> IBADataset:
    """Load a canonical dataset from HDF5."""

    return read_hdf5_dataset(path)
