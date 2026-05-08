"""Public training APIs."""

from .base import ModelTrainer, TrainingBatch, TrainingResult
from .datasets import PreparedSurrogateDataset, prepare_variable_layer_surrogate_dataset
from .losses import Chi2Loss, Log1pMSELoss
from .preprocessing import (
    ArrayTransform,
    BinMerger,
    ConstantFactorTransform,
    IdentityTransform,
    MinMaxScaler,
    ROISelector,
    SpectrumPreprocessor,
    SpectrumPreprocessorPipeline,
    split_train_val_test,
    StandardScaler,
    shuffle_in_unison,
    TrainValTestSplit,
    TransformPipeline,
)
from .supervised import EpochSchedule, SupervisedTrainer

__all__ = [
    "ArrayTransform",
    "BinMerger",
    "Chi2Loss",
    "ConstantFactorTransform",
    "EpochSchedule",
    "IdentityTransform",
    "Log1pMSELoss",
    "MinMaxScaler",
    "ModelTrainer",
    "PreparedSurrogateDataset",
    "ROISelector",
    "SpectrumPreprocessor",
    "SpectrumPreprocessorPipeline",
    "StandardScaler",
    "SupervisedTrainer",
    "TrainingBatch",
    "TrainingResult",
    "TrainValTestSplit",
    "TransformPipeline",
    "prepare_variable_layer_surrogate_dataset",
    "shuffle_in_unison",
    "split_train_val_test",
]
