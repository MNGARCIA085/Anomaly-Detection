

class PreprocessingPipeline:

    def __init__(self, steps):
        self.steps = steps

    def fit(self, X):
        for step in self.steps: # step can be scaler transform, feature selector....
            X = step.fit_transform(X)
        return self

    def transform(self, X):
        for step in self.steps:
            X = step.transform(X)
        return X

    def fit_transform_split(self, X_train, X_val):
        self.fit(X_train) # fit and transform!
        return self.transform(X_train), self.transform(X_val)

    def get_artifacts(self):
        artifacts = {}
        for i, step in enumerate(self.steps):
            if hasattr(step, "get_artifacts"):
                artifacts[f"step_{i}_{step.__class__.__name__}"] = step.get_artifacts()
        return artifacts




"""
self.fit(X_train) updates internal state (e.g. scaler mean/std)
That state is stored inside each step (e.g. self.scaler)
Then transform(X_val) uses the same fitted object

fit() → learn parameters (mean, PCA components, etc.)
transform() → apply them
Pipeline stores stateful objects
Validation/test → only transform()
"""



"""
Pieplien inf. usage
scaler = ScalerTransform.from_artifacts(saved_scaler)
selector = FeatureSelector.from_artifacts(saved_selector)

prep = PreprocessingPipeline([scaler, selector])

X_new_prep = prep.transform(X_new)

--
train
prep.fit(X_train)
artifacts = prep.get_artifacts()

inf
prep = PreprocessingPipeline.from_artifacts(artifacts)
X_new = prep.transform(X)

@classmethod
def from_artifacts(cls, artifacts):
    steps = []

    for name, step_artifacts in artifacts.items():

        if "ScalerTransform" in name:
            steps.append(ScalerTransform.from_artifacts(step_artifacts))

        elif "FeatureSelector" in name:
            steps.append(FeatureSelector.from_artifacts(step_artifacts))

    return cls(steps)


or save class name with artfs.



I also need to be able to pass them via MLFlow


"""