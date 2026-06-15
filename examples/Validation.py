from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import sys 
sys.path.append("../")

from ibamlkit.data import load_dataset, save_dataset
from ibamlkit.schema import DatasetInputSpec, IBADataset

if TYPE_CHECKING:
    from ibamlkit.generation import SIMNRASpectrumGenerator


METHODS = "NRA"
NTHREADS = 120
REPORT_EVERY = 10000
PROGRESS_EVERY = 10000
SIMNRA_RETRY_LIMIT = 3
ALLOW_FAILED_SAMPLES = True
PRINT_GENERATOR_PROGRESS = True


def collect_input_files(input_root: Path) -> list[Path]:
    root = Path(input_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Input root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Input root is not a directory: {root}")
    paths = sorted(path for path in root.rglob("*.h5") if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No .h5 files found below: {root}")
    return paths


def resolve_selected_methods(dataset: IBADataset, methods_argument: str) -> list[str]:
    available_methods = [method.name for method in dataset.input_spec.methods]
    token = methods_argument.strip()
    if not token:
        raise ValueError("methods must not be empty.")
    if token.lower() == "all":
        return available_methods

    selected_methods = [part.strip() for part in token.split(",") if part.strip()]
    if not selected_methods:
        raise ValueError("methods did not resolve to any method names.")

    unknown_methods = [name for name in selected_methods if name not in available_methods]
    if unknown_methods:
        raise ValueError(
            f"Requested methods are not present in the dataset: {unknown_methods}. "
            f"Available methods: {available_methods}"
        )
    return selected_methods


def subset_input_spec(input_spec: DatasetInputSpec, selected_methods: list[str]) -> DatasetInputSpec:
    selected_method_set = set(selected_methods)
    methods = [method for method in input_spec.methods if method.name in selected_method_set]
    parameters = [
        parameter
        for parameter in input_spec.parameters
        if not parameter.method or parameter.method in selected_method_set
    ]
    return DatasetInputSpec(
        methods=methods,
        layer_species=input_spec.layer_species,
        parameters=parameters,
        generation_info=input_spec.generation_info,
        format_version=input_spec.format_version,
    )


def subset_dataset_methods(dataset: IBADataset, selected_methods: list[str]) -> IBADataset:
    reduced_input_spec = subset_input_spec(dataset.input_spec, selected_methods)
    reduced_spectra = {
        method_name: np.asarray(dataset.spectra[method_name], dtype=np.float32)
        for method_name in selected_methods
    }
    reduced_lengths = None
    if dataset.spectra_lengths is not None:
        reduced_lengths = {
            method_name: np.asarray(dataset.spectra_lengths[method_name], dtype=np.int32)
            for method_name in selected_methods
        }
    return IBADataset(
        input_spec=reduced_input_spec,
        open_parameter_values=np.asarray(dataset.open_parameter_values, dtype=np.float32),
        spectra=reduced_spectra,
        spectra_lengths=reduced_lengths,
        sample_ids=None if dataset.sample_ids is None else list(dataset.sample_ids),
    )


def effective_length(lengths_dict, method_name: str, index: int, padded_width: int) -> int:
    if lengths_dict is None:
        return int(padded_width)
    return int(lengths_dict[method_name][index])


def padded_vector(row: np.ndarray, length: int, target_length: int) -> np.ndarray:
    out = np.zeros((target_length,), dtype=np.float64)
    if length > 0:
        out[:length] = np.asarray(row[:length], dtype=np.float64)
    return out


def compare_one_spectrum(
    old_row: np.ndarray,
    old_length: int,
    new_row: np.ndarray,
    new_length: int,
) -> tuple[float, int]:
    compare_length = max(int(old_length), int(new_length))
    old_vec = padded_vector(old_row, int(old_length), compare_length)
    new_vec = padded_vector(new_row, int(new_length), compare_length)
    chi2_vector = (old_vec - new_vec) ** 2 / (old_vec + 1.0)
    return float(np.mean(chi2_vector)), compare_length


class ValidationRunStats:
    def __init__(self, total_samples: int, report_every: int):
        self.total_samples = int(total_samples)
        self.report_every = max(0, int(report_every))
        self.failed_samples = 0
        self.last_completed = 0
        self._next_report = self.report_every if self.report_every > 0 else None
        self._lock = threading.Lock()

    def on_progress(self, progress) -> None:
        if progress.phase != "spectra":
            return
        with self._lock:
            self.last_completed = int(progress.completed)
            if self._next_report is None or self.last_completed < self._next_report:
                return
            validated = self.last_completed - self.failed_samples
            print(
                f"  progress: processed={self.last_completed}/{self.total_samples}, "
                f"validated={validated}, failed_or_dropped={self.failed_samples}"
            )
            while self._next_report is not None and self.last_completed >= self._next_report:
                self._next_report += self.report_every

    def on_error(self, failure) -> None:
        del failure
        with self._lock:
            self.failed_samples += 1

    def print_final(self, validated_sample_count: int) -> None:
        print(
            f"  file summary: processed={self.last_completed}/{self.total_samples}, "
            f"validated={validated_sample_count}, failed_or_dropped={self.failed_samples}"
        )


def regenerate_dataset(
    dataset: IBADataset,
    *,
    selected_methods: list[str],
    nthreads: int,
    report_every: int,
    progress_every: int,
    simnra_retry_limit: int,
    allow_failed_samples: bool,
    print_progress: bool,
) -> IBADataset:
    from ibamlkit.generation import SIMNRASpectrumGenerator

    dataset = subset_dataset_methods(dataset, selected_methods)
    total_samples = dataset.sample_count
    stats = ValidationRunStats(total_samples=total_samples, report_every=report_every)
    generator = SIMNRASpectrumGenerator(
        input_spec=dataset.input_spec,
        max_workers=max(1, int(nthreads)),
        keep_alive=True,
        progress_callback=stats.on_progress,
        error_callback=stats.on_error,
        progress_every=max(1, int(progress_every)),
        print_progress=bool(print_progress),
        simnra_retry_limit=max(0, int(simnra_retry_limit)),
        allow_failed_samples=bool(allow_failed_samples),
        validation_enabled=False,
    )
    try:
        sample_ids = (
            list(dataset.sample_ids)
            if dataset.sample_ids is not None
            else [f"sample-{index:08d}" for index in range(total_samples)]
        )
        validated = generator.generate(
            np.asarray(dataset.open_parameter_values, dtype=np.float32),
            sample_ids=sample_ids,
        )
        stats.print_final(validated.sample_count)
        return validated
    finally:
        generator.close()


def print_comparison_summary(original: IBADataset, validated: IBADataset) -> None:
    if original.sample_ids is None or validated.sample_ids is None:
        print("  comparison skipped: sample_ids are required on both datasets")
        return

    old_index_by_id = {str(sample_id): index for index, sample_id in enumerate(original.sample_ids)}
    new_index_by_id = {str(sample_id): index for index, sample_id in enumerate(validated.sample_ids)}
    common_ids = [sample_id for sample_id in original.sample_ids if str(sample_id) in new_index_by_id]
    if not common_ids:
        print("  comparison skipped: no overlapping sample_ids between original and validated datasets")
        return

    print(f"  compared {len(common_ids)}/{original.sample_count} original samples")
    missing = original.sample_count - len(common_ids)
    if missing:
        print(f"  dropped during validation: {missing}")

    for method in original.input_spec.methods:
        chi2_values = np.full((len(common_ids),), np.nan, dtype=np.float64)
        length_mismatches = 0
        for out_index, sample_id in enumerate(common_ids):
            old_index = old_index_by_id[str(sample_id)]
            new_index = new_index_by_id[str(sample_id)]
            old_matrix = np.asarray(original.spectra[method.name], dtype=np.float32)
            new_matrix = np.asarray(validated.spectra[method.name], dtype=np.float32)
            old_length = effective_length(
                original.spectra_lengths,
                method.name,
                old_index,
                old_matrix.shape[1],
            )
            new_length = effective_length(
                validated.spectra_lengths,
                method.name,
                new_index,
                new_matrix.shape[1],
            )
            if old_length != new_length:
                length_mismatches += 1
            chi2_values[out_index], _ = compare_one_spectrum(
                old_matrix[old_index],
                old_length,
                new_matrix[new_index],
                new_length,
            )

        print(
            f"  {method.name}: mean={float(np.nanmean(chi2_values)):.6f}, "
            f"median={float(np.nanmedian(chi2_values)):.6f}, "
            f"max={float(np.nanmax(chi2_values)):.6f}, "
            f"length_mismatches={length_mismatches}"
        )


def main() -> None:
    input_root = Path(r"D:\IBAMLKit\examples\datasets\multilayer_7_elements_seed_1").resolve()
    output_root = Path(r"D:\IBAMLKit\examples\datasets\multilayer_7_elements_seed_1_validated_NRA").resolve()
    paths = collect_input_files(input_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Input root: {input_root}")
    print(f"Output root: {output_root}")
    print(f"Found {len(paths)} file(s)")

    for file_index, input_path in enumerate(paths, start=1):
        relative_path = input_path.relative_to(input_root)
        output_path = output_root / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print()
        print(f"[{file_index}/{len(paths)}] {relative_path}")
        dataset = load_dataset(str(input_path))
        selected_methods = resolve_selected_methods(dataset, METHODS)
        print(
            f"  samples={dataset.sample_count}, "
            f"methods={selected_methods}, "
            f"layers={dataset.input_spec.generation_info.get('n_layers', 'unknown')}"
        )
        validated = regenerate_dataset(
            dataset,
            selected_methods=selected_methods,
            nthreads=NTHREADS,
            report_every=REPORT_EVERY,
            progress_every=PROGRESS_EVERY,
            simnra_retry_limit=SIMNRA_RETRY_LIMIT,
            allow_failed_samples=ALLOW_FAILED_SAMPLES,
            print_progress=PRINT_GENERATOR_PROGRESS,
        )
        print_comparison_summary(subset_dataset_methods(dataset, selected_methods), validated)
        save_dataset(str(output_path), validated)
        print(f"  wrote {output_path}")


if __name__ == "__main__":
    main()
