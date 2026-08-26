#pytest tests/unit/data/test_windowing.py


import numpy as np
import pytest

from anomaly_detection.data.windowing import Windowing


class TestWindowingInit:

    def test_rejects_seq_len_less_than_one(self):
        with pytest.raises(ValueError, match="seq_len must be >= 1"):
            Windowing(seq_len=0)

    def test_rejects_stride_less_than_one(self):
        with pytest.raises(ValueError, match="stride must be >= 1"):
            Windowing(seq_len=3, stride=0)


class TestWindowingTransform:

    def test_creates_expected_windows(self):
        X = np.array([
            [1, 10],
            [2, 20],
            [3, 30],
            [4, 40],
        ])

        windowing = Windowing(seq_len=2)

        windows = windowing.transform(X)

        expected = np.array([
            [
                [1, 10],
                [2, 20],
            ],
            [
                [2, 20],
                [3, 30],
            ],
            [
                [3, 30],
                [4, 40],
            ],
        ])

        np.testing.assert_array_equal(windows, expected)

    def test_respects_stride(self):
        X = np.arange(10).reshape(-1, 1)

        windowing = Windowing(seq_len=3, stride=2)

        windows = windowing.transform(X)

        expected = np.array([
            [[0], [1], [2]],
            [[2], [3], [4]],
            [[4], [5], [6]],
            [[6], [7], [8]],
        ])

        np.testing.assert_array_equal(windows, expected)

    def test_seq_len_one_creates_one_window_per_sample(self):
        X = np.array([
            [1, 10],
            [2, 20],
            [3, 30],
        ])

        windowing = Windowing(seq_len=1)

        windows = windowing.transform(X)

        assert windows.shape == (3, 1, 2)

        expected = np.array([
            [[1, 10]],
            [[2, 20]],
            [[3, 30]],
        ])

        np.testing.assert_array_equal(windows, expected)

    def test_rejects_non_2d_input(self):
        X = np.array([1, 2, 3])

        windowing = Windowing(seq_len=2)

        with pytest.raises(ValueError, match="Expected X with shape"):
            windowing.transform(X)

    def test_rejects_input_shorter_than_sequence_length(self):
        X = np.array([
            [1, 10],
            [2, 20],
        ])

        windowing = Windowing(seq_len=3)

        with pytest.raises(ValueError, match="Not enough samples"):
            windowing.transform(X)


class TestWindowingWithLabels:

    def test_window_is_anomalous_if_any_sample_is_anomalous(self):
        X = np.array([
            [1],
            [2],
            [3],
            [4],
        ])

        y = np.array([0, 0, 1, 0])

        windowing = Windowing(seq_len=2)

        X_windows, y_windows = windowing.transform_with_labels(X, y)

        expected_X = np.array([
            [[1], [2]],
            [[2], [3]],
            [[3], [4]],
        ])

        expected_y = np.array([0, 1, 1])

        np.testing.assert_array_equal(X_windows, expected_X)
        np.testing.assert_array_equal(y_windows, expected_y)

    def test_rejects_x_y_with_different_lengths(self):
        X = np.array([
            [1],
            [2],
            [3],
        ])

        y = np.array([0, 1])

        windowing = Windowing(seq_len=2)

        with pytest.raises(
            ValueError,
            match="X and y must have the same number of samples",
        ):
            windowing.transform_with_labels(X, y)