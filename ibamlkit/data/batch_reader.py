"""Batch reading helpers for one dataset file or many compatible shards."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..schema import DatasetInputSpec, IBADataset
from .io import load_dataset


class DatasetBatchReader:
    """Read one dataset file or concatenate many compatible dataset shards."""

    def collect_dataset_paths(self, input_path: Path) -> list[Path]:
        """Return one dataset path or all ``.h5`` shard paths in a directory."""

        if input_path.is_file():
            return [input_path]
        if input_path.is_dir():
            paths = sorted(input_path.glob("*.h5"))
            if not paths:
                raise FileNotFoundError(f"No .h5 files found in directory: {input_path}")
            return paths
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    def assert_compatible_specs(
        self,
        base: DatasetInputSpec,
        others: list[DatasetInputSpec],
    ) -> None:
        """Check that shard specs can be concatenated safely."""

        base_open_names = [parameter.name for parameter in base.open_parameters]
        base_method_names = [method.name for method in base.methods]

        for index, other in enumerate(others, start=2):
            other_open_names = [parameter.name for parameter in other.open_parameters]
            other_method_names = [method.name for method in other.methods]
            if other_open_names != base_open_names:
                raise ValueError(
                    f"Dataset #{index} has different open-parameter ordering and cannot be concatenated."
                )
            if other_method_names != base_method_names:
                raise ValueError(
                    f"Dataset #{index} has different method ordering and cannot be concatenated."
                )

    def concatenate_padded_spectra(self, spectra_blocks: list[np.ndarray]) -> np.ndarray:
        """Pad shard-local spectra widths to a global width and concatenate them."""

        if not spectra_blocks:
            return np.zeros((0, 0), dtype=np.float32)

        max_width = max(block.shape[1] for block in spectra_blocks)
        padded_blocks = []
        for block in spectra_blocks:
            block = np.asarray(block, dtype=np.float32)
            if block.shape[1] == max_width:
                padded_blocks.append(block)
                continue
            padded = np.zeros((block.shape[0], max_width), dtype=np.float32)
            padded[:, : block.shape[1]] = block
            padded_blocks.append(padded)
        return np.concatenate(padded_blocks, axis=0)

    def load_many(self, paths: list[Path]) -> IBADataset:
        """Load one or more dataset files and concatenate them."""

        datasets = [load_dataset(str(path)) for path in paths]
        if not datasets:
            raise ValueError("No dataset files were loaded.")

        base_spec = datasets[0].input_spec
        self.assert_compatible_specs(base_spec, [dataset.input_spec for dataset in datasets[1:]])

        open_parameter_values = np.concatenate(
            [dataset.open_parameter_values for dataset in datasets],
            axis=0,
        )

        spectra = {
            method.name: self.concatenate_padded_spectra(
                [dataset.spectra[method.name] for dataset in datasets]
            )
            for method in base_spec.methods
        }

        spectra_lengths = None
        if all(dataset.spectra_lengths is not None for dataset in datasets):
            spectra_lengths = {
                method.name: np.concatenate(
                    [dataset.spectra_lengths[method.name] for dataset in datasets],
                    axis=0,
                )
                for method in base_spec.methods
            }

        sample_ids = None
        if all(dataset.sample_ids is not None for dataset in datasets):
            sample_ids = []
            for dataset in datasets:
                sample_ids.extend(dataset.sample_ids)

        return IBADataset(
            input_spec=base_spec,
            open_parameter_values=open_parameter_values,
            spectra=spectra,
            spectra_lengths=spectra_lengths,
            sample_ids=sample_ids,
        )

    def print_summary(self, dataset: IBADataset, paths: list[Path]) -> None:
        """Print a short human-readable summary."""

        print(f"Loaded {len(paths)} file(s)")
        print(f"Samples: {dataset.sample_count}")
        print(f"Methods: {[method.name for method in dataset.input_spec.methods]}")
        print(
            f"Open parameters: {[parameter.name for parameter in dataset.input_spec.open_parameters]}"
        )
        print(f"Generation info: {dict(dataset.input_spec.generation_info)}")

        for method in dataset.input_spec.methods:
            shape = dataset.spectra[method.name].shape
            print(f"Spectrum matrix for {method.name}: shape={shape}")
            if dataset.spectra_lengths is not None:
                lengths = dataset.spectra_lengths[method.name]
                print(
                    f"  true lengths: min={int(lengths.min())}, max={int(lengths.max())}, "
                    f"mean={float(lengths.mean()):.1f}"
                )
