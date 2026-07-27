class PreprocessingPipeline:

    def __init__(self, steps):
        self.steps = steps

    def fit(self, X):

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