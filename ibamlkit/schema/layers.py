"""Schema objects describing the shared layer layout of a dataset."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerSpeciesSpec:
    """One element or isotope entry declared in a physical layer."""

    layer_index: int
    element: str
    isotope: str = ""

    def __post_init__(self) -> None:
        if self.layer_index < 1:
            raise ValueError("Layer indices are 1-based and must be >= 1.")
        if not self.element:
            raise ValueError("Element name must not be empty.")
