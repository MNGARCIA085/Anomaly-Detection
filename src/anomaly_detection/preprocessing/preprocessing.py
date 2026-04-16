from abc import ABC



# base
class Transform(ABC):

    def fit(self, X):
        return self

    def transform(self, X):
        return X


    # each subclass should define whats worth saving
    def get_artifacts(self):
        raise NotImplementedError


    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    









# reusable pieces
class ScalerTransform(Transform):

    def __init__(self, scaler_cls):
        self.scaler = scaler_cls()

    def fit(self, X):
        self.scaler.fit(X)
        return self

    def transform(self, X):
        return self.scaler.transform(X)


    def get_artifacts(self):
    	return self.scaler

    """
    def get_artifacts(self):
    return {
        "type": "scaler",
        "object": self.scaler
    }
    """




class FeatureSelector(Transform):

    def __init__(self, indices):
        self.indices = indices


    def fit(self, X):
        return self

    def transform(self, X):
        return X[:, self.indices]


    # EACH class defines whats worth saving
    def get_artifacts(self):
       return {"indices": self.indices}


"""
later
PCA
Log transform
Clipping
Custom domain transform
"""



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
                artifacts[f"step_{i}_{step.__class__.__name__}"] = step.get_artifact()
        return artifacts


"""
self.fit(X_train) updates internal state (e.g. scaler mean/std)
That state is stored inside each step (e.g. self.scaler)
Then transform(X_val) uses the same fitted object
"""


"""
fit() → learn parameters (mean, PCA components, etc.)
transform() → apply them
Pipeline stores stateful objects
Validation/test → only transform()
"""



"""
examples

prep = PreprocessingPipeline([
    ScalerTransform(StandardScaler)
])

prep = PreprocessingPipeline([
    ScalerTransform(StandardScaler),
    FeatureSelector([0,1,2,3,4,5])
])

prep = PreprocessingPipeline([])
"""


"""
later: add builder

def build_preprocessor(cfg):

    steps = []

    if cfg.use_scaler:
        steps.append(ScalerTransform(cfg.scaler_cls))

    if cfg.feature_indices:
        steps.append(FeatureSelector(cfg.feature_indices))

    return PreprocessingPipeline(steps)

"""
