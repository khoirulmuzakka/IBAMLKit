"""Sampling helpers for dataset generation.

This module focuses on generating open-parameter matrices for the
``DatasetInputSpec`` schema, with special handling for layer concentration
parameters.

Why concentration sampling is special
------------------------------------

For one layer, the elemental concentrations must sum to a fixed total:

- usually ``1.0`` for a fully open layer
- or ``1.0 - fixed_total`` when some layer concentrations are fixed

Naively drawing positive random numbers and normalizing them produces points
that cluster toward small values when the number of elements grows. In
practical IBA this is often undesirable because:

- pure species should be represented explicitly
- one- or two-dominant compositions are common
- high-concentration cases for each element become too rare under uniform
  simplex sampling

Sampling strategy
-----------------

The concentration sampler therefore mixes two sources of samples:

1. Deterministic anchors
   Hand-crafted concentration vectors that guarantee important edge cases are
   present in the dataset.

2. Random dominance-pattern sampling
   Stochastic samples that are biased toward realistic IBA-like compositions.

Deterministic anchors
---------------------

The anchor library is built once per layer size and may contain:

- pure species vectors, for example ``[1, 0, 0, 0]``
- pair anchors, when enabled:
  - ``50/50``
  - ``70/30``
  - ``30/70``

The parameter ``anchor_fraction`` controls what fraction of requested samples
is taken from this anchor library. The anchor vectors themselves are
deterministic; the only randomness is which anchors are selected for the
current batch.

Random modes
------------

The remaining samples are drawn from one of several random modes:

- ``pure``
- ``single_dominant``
- ``double_dominant``
- ``triple_dominant``
- ``sparse_tail``
- ``balanced``

These mode weights are controlled by ``LayerConcentrationSamplingConfig``.
They are normalized within the non-anchor portion only.

Important note on pure species
------------------------------

At the moment, pure species can appear twice:

- via deterministic anchors
- via the random ``pure`` mode

This gives strong pure-species coverage, but it also means pure cases are
double-counted conceptually. If stricter control of fractions is desired,
set ``pure_weight=0.0`` and keep pure species only in the anchor library.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping

import numpy as np

from ..schema import DatasetInputSpec, ParameterSpec


@dataclass(frozen=True)
class LayerConcentrationSamplingConfig:
    """Sampling configuration for one layer's concentration simplex.

    The generated dataset is split into:

    - an anchor portion controlled by ``anchor_fraction``
    - a random portion controlled by the mode weights below

    The random weights are normalized among themselves; they do not include
    the anchor fraction.
    """

    anchor_fraction: float = 0.2
    pure_weight: float = 0.1
    single_dominant_weight: float = 0.35
    double_dominant_weight: float = 0.3
    triple_dominant_weight: float = 0.1
    sparse_tail_weight: float = 0.1
    balanced_weight: float = 0.05
    dominant_threshold: float = 0.2
    max_minor_active: int = 3
    include_pair_anchors: bool = True


def sample_open_parameter_matrix(
    input_spec: DatasetInputSpec,
    n_samples: int,
    seed: int | None = None,
    layer_sampling: Mapping[int, LayerConcentrationSamplingConfig] | None = None,
) -> np.ndarray:
    """Sample open parameters from bounds with concentration-aware layer sampling.

    Non-concentration open parameters are sampled independently and uniformly
    from their declared bounds.

    Open concentration parameters are sampled layer by layer:

    - fixed concentration parameters in that layer consume part of the total
      concentration budget
    - the remaining open concentration budget is sampled using
      :func:`sample_layer_concentrations`

    The returned matrix matches ``input_spec.open_parameters`` column order.
    """

    rng = np.random.default_rng(seed)
    open_parameters = list(input_spec.open_parameters)
    sampled = np.zeros((n_samples, len(open_parameters)), dtype=np.float32)

    for column, parameter in enumerate(open_parameters):
        if _is_concentration_parameter(parameter):
            continue
        if parameter.lower_bound is None or parameter.upper_bound is None:
            raise ValueError(f"Open parameter '{parameter.name}' must define bounds.")
        sampled[:, column] = rng.uniform(
            low=parameter.lower_bound,
            high=parameter.upper_bound,
            size=n_samples,
        )

    layer_sampling = dict(layer_sampling or {})
    for layer_index, columns in _open_concentration_columns(open_parameters).items():
        config = layer_sampling.get(layer_index, LayerConcentrationSamplingConfig())
        fixed_total = _fixed_concentration_total(input_spec, layer_index)
        free_budget = 1.0 - fixed_total
        if free_budget <= 0.0:
            raise ValueError(f"Layer {layer_index} has no free concentration budget.")
        sampled[:, columns] = sample_layer_concentrations(
            n_elements=len(columns),
            n_samples=n_samples,
            total=free_budget,
            rng=rng,
            config=config,
        )

    return sampled


def sample_layer_concentrations(
    n_elements: int,
    n_samples: int,
    total: float,
    rng: np.random.Generator,
    config: LayerConcentrationSamplingConfig | None = None,
) -> np.ndarray:
    """Sample concentration vectors that sum to ``total``.

    Parameters
    ----------
    n_elements:
        Number of open concentration components in the layer.
    n_samples:
        Number of concentration vectors to generate.
    total:
        The total concentration budget for the open components. This is
        normally ``1.0`` for a fully open layer, or ``1.0 - fixed_total`` if
        some concentrations are fixed elsewhere in the same layer.
    rng:
        Random number generator.
    config:
        Sampling policy controlling anchors and random modes.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_samples, n_elements)`` whose rows sum to ``total``.
    """

    if n_elements <= 0:
        return np.zeros((n_samples, 0), dtype=np.float32)
    if total <= 0.0:
        raise ValueError("Concentration total must be positive.")

    cfg = config or LayerConcentrationSamplingConfig()
    samples = np.zeros((n_samples, n_elements), dtype=np.float32)

    anchors = _build_anchor_library(n_elements=n_elements, total=total, config=cfg)
    n_anchor = min(n_samples, len(anchors), int(round(cfg.anchor_fraction * n_samples)))
    if n_anchor > 0:
        chosen = rng.choice(len(anchors), size=n_anchor, replace=False)
        samples[:n_anchor] = anchors[chosen]

    if n_anchor == n_samples:
        return samples

    mode_names = np.array(
        [
            "pure",
            "single_dominant",
            "double_dominant",
            "triple_dominant",
            "sparse_tail",
            "balanced",
        ],
        dtype=object,
    )
    weights = np.array(
        [
            cfg.pure_weight,
            cfg.single_dominant_weight,
            cfg.double_dominant_weight,
            cfg.triple_dominant_weight,
            cfg.sparse_tail_weight,
            cfg.balanced_weight,
        ],
        dtype=np.float64,
    )
    if np.all(weights <= 0.0):
        raise ValueError("At least one sampling mode must have positive weight.")
    weights = weights / weights.sum()

    for row_index in range(n_anchor, n_samples):
        mode = str(rng.choice(mode_names, p=weights))
        samples[row_index] = _sample_mode(
            mode=mode,
            n_elements=n_elements,
            total=total,
            rng=rng,
            config=cfg,
        )

    return samples


def _is_concentration_parameter(parameter: ParameterSpec) -> bool:
    return parameter.group == "layer" and parameter.kind == "concentration"


def _open_concentration_columns(open_parameters: list[ParameterSpec]) -> dict[int, list[int]]:
    ret: dict[int, list[int]] = {}
    for index, parameter in enumerate(open_parameters):
        if _is_concentration_parameter(parameter):
            ret.setdefault(parameter.layer_index, []).append(index)
    return ret


def _fixed_concentration_total(input_spec: DatasetInputSpec, layer_index: int) -> float:
    total = 0.0
    for parameter in input_spec.fixed_parameters:
        if _is_concentration_parameter(parameter) and parameter.layer_index == layer_index:
            total += float(parameter.fixed_value)
    return total


def _build_anchor_library(
    n_elements: int,
    total: float,
    config: LayerConcentrationSamplingConfig,
) -> np.ndarray:
    """Construct deterministic anchor vectors for a layer.

    The library currently contains:

    - all pure species anchors
    - optional pair anchors: ``50/50``, ``70/30``, and ``30/70``

    These anchors are later sampled without replacement for the anchor
    portion of a generated batch.
    """
    rows: list[np.ndarray] = []
    eye = np.eye(n_elements, dtype=np.float32)
    for index in range(n_elements):
        rows.append(total * eye[index])

    if config.include_pair_anchors and n_elements >= 2:
        for left, right in combinations(range(n_elements), 2):
            pair = np.zeros(n_elements, dtype=np.float32)
            pair[left] = 0.5 * total
            pair[right] = 0.5 * total
            rows.append(pair)

            skew_lr = np.zeros(n_elements, dtype=np.float32)
            skew_lr[left] = 0.7 * total
            skew_lr[right] = 0.3 * total
            rows.append(skew_lr)

            skew_rl = np.zeros(n_elements, dtype=np.float32)
            skew_rl[left] = 0.3 * total
            skew_rl[right] = 0.7 * total
            rows.append(skew_rl)

    if not rows:
        return np.zeros((0, n_elements), dtype=np.float32)
    return np.stack(rows, axis=0)


def _sample_mode(
    mode: str,
    n_elements: int,
    total: float,
    rng: np.random.Generator,
    config: LayerConcentrationSamplingConfig,
) -> np.ndarray:
    """Dispatch one random concentration sample to the requested mode."""
    if mode == "pure":
        return _sample_pure(n_elements, total, rng)
    if mode == "single_dominant":
        return _sample_single_dominant(n_elements, total, rng, config)
    if mode == "double_dominant":
        return _sample_double_dominant(n_elements, total, rng, config)
    if mode == "triple_dominant":
        return _sample_triple_dominant(n_elements, total, rng, config)
    if mode == "sparse_tail":
        return _sample_sparse_tail(n_elements, total, rng, config)
    return _sample_balanced(n_elements, total, rng)


def _sample_pure(n_elements: int, total: float, rng: np.random.Generator) -> np.ndarray:
    """Sample a one-hot pure-species concentration vector."""
    row = np.zeros(n_elements, dtype=np.float32)
    row[int(rng.integers(0, n_elements))] = total
    return row


def _sample_single_dominant(
    n_elements: int,
    total: float,
    rng: np.random.Generator,
    config: LayerConcentrationSamplingConfig,
) -> np.ndarray:
    """Sample a composition with exactly one dominant element.

    One element is forced above ``dominant_threshold``. The remaining budget,
    if any, is distributed over a sparse tail of minor species.
    """
    if total <= config.dominant_threshold:
        return _sample_balanced(n_elements, total, rng)

    dominant = int(rng.integers(0, n_elements))
    dominant_value = float(rng.uniform(config.dominant_threshold, total))
    remainder = total - dominant_value

    row = np.zeros(n_elements, dtype=np.float32)
    row[dominant] = dominant_value
    if remainder > 0.0:
        minor_indices = [index for index in range(n_elements) if index != dominant]
        _fill_minor_tail(row, minor_indices, remainder, rng, config)
    return row


def _sample_double_dominant(
    n_elements: int,
    total: float,
    rng: np.random.Generator,
    config: LayerConcentrationSamplingConfig,
) -> np.ndarray:
    """Sample a composition with exactly two dominant elements.

    Both dominant elements are forced above ``dominant_threshold`` and the
    remaining budget is split between them and an optional sparse minor tail.
    """
    if n_elements < 2 or total <= 2.0 * config.dominant_threshold:
        return _sample_single_dominant(n_elements, total, rng, config)

    dominant_indices = rng.choice(n_elements, size=2, replace=False)
    base = np.full(2, config.dominant_threshold, dtype=np.float64)
    extra_budget = total - float(base.sum())
    split = rng.dirichlet(np.ones(2))
    dominant_values = base + extra_budget * split
    remainder = total - float(dominant_values.sum())

    row = np.zeros(n_elements, dtype=np.float32)
    row[dominant_indices] = dominant_values.astype(np.float32)
    if remainder > 0.0:
        minor_indices = [index for index in range(n_elements) if index not in dominant_indices]
        _fill_minor_tail(row, minor_indices, remainder, rng, config)
    return row


def _sample_triple_dominant(
    n_elements: int,
    total: float,
    rng: np.random.Generator,
    config: LayerConcentrationSamplingConfig,
) -> np.ndarray:
    """Sample a composition with exactly three dominant elements.

    This is only feasible when the concentration budget can support three
    components above the dominance threshold.
    """
    if n_elements < 3 or total <= 3.0 * config.dominant_threshold:
        return _sample_double_dominant(n_elements, total, rng, config)

    dominant_indices = rng.choice(n_elements, size=3, replace=False)
    base = np.full(3, config.dominant_threshold, dtype=np.float64)
    extra_budget = total - float(base.sum())
    split = rng.dirichlet(np.ones(3))
    dominant_values = base + extra_budget * split
    remainder = total - float(dominant_values.sum())

    row = np.zeros(n_elements, dtype=np.float32)
    row[dominant_indices] = dominant_values.astype(np.float32)
    if remainder > 0.0:
        minor_indices = [index for index in range(n_elements) if index not in dominant_indices]
        _fill_minor_tail(row, minor_indices, remainder, rng, config)
    return row


def _sample_sparse_tail(
    n_elements: int,
    total: float,
    rng: np.random.Generator,
    config: LayerConcentrationSamplingConfig,
) -> np.ndarray:
    """Sample a sparse composition with one or two dominant species.

    Compared with the other dominant samplers, this mode explicitly tries to
    keep only a small number of minor species active.
    """
    if n_elements == 1:
        return np.asarray([total], dtype=np.float32)

    dominant_count = 1 if n_elements < 3 else int(rng.choice([1, 2], p=[0.6, 0.4]))
    if dominant_count == 2 and total <= 2.0 * config.dominant_threshold:
        dominant_count = 1

    dominant_indices = rng.choice(n_elements, size=dominant_count, replace=False)
    row = np.zeros(n_elements, dtype=np.float32)

    if dominant_count == 1:
        dom_value = float(rng.uniform(config.dominant_threshold, total))
        row[dominant_indices[0]] = dom_value
        remainder = total - dom_value
    else:
        base = np.full(dominant_count, config.dominant_threshold, dtype=np.float64)
        extra_budget = total - float(base.sum())
        split = rng.dirichlet(np.ones(dominant_count))
        dominant_values = base + extra_budget * split
        row[dominant_indices] = dominant_values.astype(np.float32)
        remainder = total - float(dominant_values.sum())

    if remainder > 0.0:
        minor_indices = [index for index in range(n_elements) if index not in dominant_indices]
        _fill_minor_tail(row, minor_indices, remainder, rng, config)
    return row


def _sample_balanced(n_elements: int, total: float, rng: np.random.Generator) -> np.ndarray:
    """Sample a fully dense composition on the simplex."""
    return (total * rng.dirichlet(np.ones(n_elements))).astype(np.float32)


def _fill_minor_tail(
    row: np.ndarray,
    minor_indices: list[int],
    remainder: float,
    rng: np.random.Generator,
    config: LayerConcentrationSamplingConfig,
) -> None:
    """Distribute the remaining budget over a sparse set of minor species."""
    if not minor_indices or remainder <= 0.0:
        return

    max_active = min(len(minor_indices), max(1, config.max_minor_active))
    active_count = int(rng.integers(1, max_active + 1))
    active = rng.choice(minor_indices, size=active_count, replace=False)
    weights = rng.dirichlet(np.ones(active_count))
    row[active] = (remainder * weights).astype(np.float32)
