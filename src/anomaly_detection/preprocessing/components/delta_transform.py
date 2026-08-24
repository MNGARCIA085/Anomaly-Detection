

class DeltaTransform:

    def fit(self, X):
        return self

    def transform(self, X):

        if X.ndim != 3:
            raise ValueError(
                "DeltaTransform expects "
                "(samples, window, features)"
            )

        return X[:, 1:, :] - X[:, :-1, :]