import unittest

import numpy as np
import torch

from ibamlkit.training import BlockShuffleBatchSampler, EpochSchedule, SupervisedTrainer


class DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)

    def transform_inputs(self, inputs):
        return torch.as_tensor(inputs, dtype=torch.float32)

    def validate_input_shape(self, inputs: torch.Tensor) -> None:
        if inputs.ndim != 2 or inputs.shape[1] != 2:
            raise ValueError("Expected 2D inputs with 2 features.")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


class SupervisedTrainerTests(unittest.TestCase):
    def test_fit_without_train_loss_tracking(self) -> None:
        model = DummyModel()
        trainer = SupervisedTrainer(
            device="cpu",
            verbose=False,
            track_train_loss=False,
            batch_shuffle_mode="block",
            shuffle_block_size=2,
            eval_batch_size=2,
        )
        train_inputs = np.asarray([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]], dtype=np.float32)
        train_targets = np.asarray([[1.0], [0.0], [0.5]], dtype=np.float32)
        val_inputs = np.asarray([[0.25, 0.75]], dtype=np.float32)
        val_targets = np.asarray([[0.75]], dtype=np.float32)

        result = trainer.fit(
            model,
            train_inputs=train_inputs,
            train_targets=train_targets,
            val_inputs=val_inputs,
            val_targets=val_targets,
            schedule=[EpochSchedule(learning_rate=1e-2, epochs=1, batch_size=2)],
        )

        self.assertIn("val_loss", result.metrics)
        self.assertNotIn("train_loss", result.metrics)
        self.assertEqual(result.epochs_completed, 1)

    def test_block_shuffle_batch_sampler_yields_contiguous_batches(self) -> None:
        sampler = BlockShuffleBatchSampler(sample_count=10, batch_size=2, block_size=4, seed=7)
        batches = list(sampler)

        self.assertTrue(all(len(batch) <= 2 for batch in batches))
        self.assertTrue(all(batch == list(range(batch[0], batch[0] + len(batch))) for batch in batches))
        self.assertEqual(sorted(index for batch in batches for index in batch), list(range(10)))


if __name__ == "__main__":
    unittest.main()
