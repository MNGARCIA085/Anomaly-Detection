import numpy as np
import torch

from anomaly_detection.training.dataset import AnomalyDataset


def test_anomaly_dataset_returns_float_tensor():
    """AnomalyDataset should convert samples to float32 tensors."""
    X = np.array([
        [1, 2],
        [3, 4],
        [5, 6],
    ])

    dataset = AnomalyDataset(X)

    assert len(dataset) == 3

    sample = dataset[0]

    assert isinstance(sample, torch.Tensor)
    assert sample.dtype == torch.float32
    assert sample.shape == (2,)

    torch.testing.assert_close(
        sample,
        torch.tensor([1.0, 2.0]),
    )