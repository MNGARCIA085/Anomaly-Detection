import numpy as np


"""
X_train.shape == (N, T, F)
X_val.shape   == (M, T, F)
y_val.shape   == (M,)
"""



class Windowing:
    def __init__(self, seq_len: int, stride: int = 1):
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        if stride < 1:
            raise ValueError("stride must be >= 1")

        self.seq_len = seq_len
        self.stride = stride

    def transform(self, X):
        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(
                f"Expected X with shape (n_samples, n_features), got {X.shape}"
            )

        n_samples = X.shape[0]

        if n_samples < self.seq_len:
            raise ValueError(
                f"Not enough samples ({n_samples}) for seq_len={self.seq_len}"
            )

        starts = range(
            0,
            n_samples - self.seq_len + 1,
            self.stride,
        )

        windows = np.stack(
            [X[i:i + self.seq_len] for i in starts]
        )

        return windows

    def transform_with_labels(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError(
                f"Expected X with shape (n_samples, n_features), got {X.shape}"
            )

        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")

        if len(X) < self.seq_len:
            raise ValueError(
                f"Not enough samples ({len(X)}) for seq_len={self.seq_len}"
            )

        X_windows = []
        y_windows = []

        for i in range(
            0,
            len(X) - self.seq_len + 1,
            self.stride,
        ):
            X_windows.append(
                X[i:i + self.seq_len]
            )

            # Window is anomalous if any point is anomalous
            y_windows.append(
                int(np.any(y[i:i + self.seq_len] == 1))
            )

        return (
            np.stack(X_windows),
            np.asarray(y_windows),
        )