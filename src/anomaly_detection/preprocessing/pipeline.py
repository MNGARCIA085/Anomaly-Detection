

class PreprocessingPipeline:

    def __init__(self, steps):
        self.steps = steps # scaler, feat. sel..... "i saved them"

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

    def get_artifacts0(self):
        artifacts = {}
        for i, step in enumerate(self.steps):
            if hasattr(step, "get_artifacts"):
                artifacts[f"step_{i}_{step.__class__.__name__}"] = step.get_artifacts()
        return artifacts

    def get_artifacts(self):
        artifacts = {}
        for i, step in enumerate(self.steps):
            step_name = f"step_{i}_{step.__class__.__name__}"
            
            # 1. If it's a custom step that has the explicit method
            if hasattr(step, "get_artifacts"):
                artifacts[step_name] = step.get_artifacts()
            
            # 2. Fallback for scikit-learn estimators (extract fitted attributes)
            else:
                # Gathers attributes like mean_, scale_, var_ 
                learned_params = {k: v for k, v in vars(step).items() if k.endswith("_") and not k.startswith("__")}
                if learned_params:
                    artifacts[step_name] = learned_params
                    
        return artifacts


    """
    maybe better
    def fit_transform_split(self, X_train, X_val):
        # fit_transform the training data in one pass
        X_train_transformed = self.fit_transform(X_train) 
        # only transform the validation data
        X_val_transformed = self.transform(X_val) 
        return X_train_transformed, X_val_transformed

    def fit_transform(self, X):
        for step in self.steps:
            X = step.fit_transform(X)
        return X
    """


# https://chatgpt.com/c/6a36d2e3-86c0-83ea-a21f-b53759a625a8



# move later to models


from sklearn.preprocessing import StandardScaler


class AEPreprocessingBuilder:

    def build(self, cfg):
        steps = []

        if cfg.scale:
            steps.append(StandardScaler())

        #if cfg.feature_selection:
        #    steps.append(AEFeatureSelector())

        return PreprocessingPipeline(steps)


class IsoForestPreprocessingBuilder:

    def build(self, cfg):
        steps = []

        if cfg.scale:
            pass
            #steps.append(RobustScaler())

        # maybe no feature selection

        return PreprocessingPipeline(steps)



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