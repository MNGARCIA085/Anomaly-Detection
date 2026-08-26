import numpy as np
import pytest

from anomaly_detection.preprocessing.temporal.delta_transform import (
    DeltaTransform,
)


class TestDeltaTransform:

    def test_fit_returns_self(self):
        """fit should follow the transformer interface and return itself."""
        X = np.zeros((2, 3, 4))

        transform = DeltaTransform()

        result = transform.fit(X)

        assert result is transform

    def test_transform_computes_deltas_between_consecutive_timesteps(self):
        """Each output timestep should be the difference from the previous one."""
        X = np.array([
            [
                [1, 10],
                [3, 15],
                [6, 25],
            ],
            [
                [10, 100],
                [15, 120],
                [20, 150],
            ],
        ])

        transform = DeltaTransform()

        result = transform.transform(X)

        expected = np.array([
            [
                [2, 5],
                [3, 10],
            ],
            [
                [5, 20],
                [5, 30],
            ],
        ])

        np.testing.assert_array_equal(result, expected)

    def test_transform_rejects_non_3d_input(self):
        """DeltaTransform should only accept windowed temporal data."""
        X = np.array([
            [1, 2],
            [3, 4],
        ])

        transform = DeltaTransform()

        with pytest.raises(
            ValueError,
            match="DeltaTransform expects",
        ):
            transform.transform(X)



"""
(samples, window, features)
          ↓
difference between consecutive timesteps
          ↓
(samples, window - 1, features)
"""