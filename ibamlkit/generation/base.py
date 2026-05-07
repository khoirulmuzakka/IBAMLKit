"""Base interfaces for dataset generators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from ..schema import IBADataset


class SpectrumGenerator(ABC):
    """Abstract interface for generators that produce spectra from open parameters."""

    @abstractmethod
    def generate(
        self,
        open_parameter_values: np.ndarray,
        sample_ids: Sequence[str] | None = None,
    ) -> IBADataset:
        """Generate a complete in-memory dataset."""

    @abstractmethod
    def generate_to_files(
        self,
        open_parameter_values: np.ndarray,
        output_dir: str,
        base_name: str,
        chunk_size: int,
        sample_ids: Sequence[str] | None = None,
    ) -> list[str]:
        """Generate one or more on-disk dataset shards."""
