import joblib


class PreprocessingPipeline:

    def __init__(self, steps):
        self.steps = steps


    def fit(self, X):

        # test remove it
        #self._input_dim = X.shape[1] feaures for 2d bu wind. length for 3D


        for step in self.steps:
            if hasattr(step, "fit"):
                step.fit(X)

            X = step.transform(X)

        return self


    def transform(self, X):

        for step in self.steps:
            X = step.transform(X)

        return X


    def fit_transform(self, X):
        return self.fit(X).transform(X)


    # for later loggign
    def save(self, path):

        joblib.dump(
            self,
            path
        )


    """
    @property
    def input_dim(self):
        if self._input_dim is None:
            raise RuntimeError(
                "Pipeline must be fitted before accessing input_dim."
            )
        return self._input_dim
    """
