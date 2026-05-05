# IBAMLKit

`IBAMLKit` is a public toolkit for machine-learning workflows in ion beam analysis (IBA).

Its main purpose is to define a reusable and community-facing standard for:

- dataset generation and storage
- model input/output conventions
- training support for surrogate and inverse models
- packaging trained models for use in analysis software such as AutoNRA, DataFurnace, and others

## Goals

`IBAMLKit` is intended to provide a stable foundation so that researchers can:

- generate reusable IBA datasets
- train PyTorch-based surrogate models
- train inverse models that infer sample parameters from spectra
- export compatible models to ONNX
- package those exported models so AutoNRA can import and use them

The long-term aim is interoperability. A model should not need to be developed inside AutoNRA to be usable by AutoNRA.

## Scope

This repository is organized around three layers:

1. Data standards and generation
2. Model APIs and training utilities
3. Deployment and packaging for analysis-software compatibility

The repository is broader than an AutoNRA SDK. Integration with analysis software such as AutoNRA and DataFurnace is an important target, but the dataset and model standards are intended to be reusable by the wider IBA/ML community.

## Core Idea

`IBAMLKit` does not define one specific neural network architecture. Instead, it defines the contract around a model:

- what the physical input representation looks like
- how spectra and labels are stored
- how surrogate and inverse tasks are represented
- how trained models are exported
- what metadata must accompany an exported model

In practice, users may train models in PyTorch, export them to ONNX, and package them according to the `IBAMLKit` standard. AutoNRA can then load that package if it supports the declared package version.

## Repository Layout

- `docs/`
  Specification and design documents.
- `ibamlkit/schema/`
  Canonical Python schema for setup parameters, layers, spectra, samples, datasets, and package metadata.
- `ibamlkit/data/`
  Dataset I/O, validation, provenance, and split utilities.
- `ibamlkit/generation/`
  Data generation pipelines, including SIMNRA-oriented tooling.
- `ibamlkit/encoding/`
  Canonical encoders and normalization logic for surrogate and inverse models.
- `ibamlkit/models/`
  Base Python APIs for model implementations.
- `ibamlkit/training/`
  Optional training and evaluation helpers.
- `ibamlkit/export/`
  ONNX export and package creation utilities.
- `ibamlkit/sdk/`
  Analysis-software-facing compatibility contracts and package validation logic.
- `examples/`
  Minimal reference workflows.
- `tests/`
  Contract and tooling tests.

## Planned Specifications

The initial repository structure reserves space for three core specifications:

- `docs/dataset_spec.md`
- `docs/model_io_spec.md`
- `docs/package_spec.md`

These documents are expected to become the normative definition of the public standard.

## Relation To Analysis Software

`IBAMLKit` is intended to provide an open and public model-development path for the community, independent of any single analysis application.

The expected workflow is:

1. Generate or collect training data using the `IBAMLKit` data standard.
2. Train a surrogate or inverse model in Python.
3. Export the trained model to ONNX.
4. Package the model with the required metadata and validation artifacts.
5. Import the packaged model into compatible analysis software.

Analysis software should only depend on the exported package contract, not on the training pipeline used to produce the model.

## Status

This repository is currently in early scaffolding stage. The folder layout has been created, but the specifications and implementation are still to be written.
