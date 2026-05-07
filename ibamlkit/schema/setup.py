"""Schema objects describing IBA method metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping
import json


@dataclass(frozen=True)
class MethodSpec:
    """Definition of one IBA method present in a dataset."""

    name: str
    reference_file: str = ""
    file_type: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Method name must not be empty.")
        json.dumps(dict(self.metadata))
