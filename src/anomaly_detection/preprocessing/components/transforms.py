import numpy as np
from sklearn.preprocessing import PowerTransformer


class FunctionTransform:

    def __init__(self, func):
        self.func = func

    def fit(self, X):
        return self

    def transform(self, X):
        return self.func(X)



TRANSFORM_REGISTRY = {
    "power": PowerTransformer,
}


def create_transform(name, **params):
    try:
        transform_cls = TRANSFORM_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown transform: {name}. "
            f"Available: {list(TRANSFORM_REGISTRY)}"
        )

    return transform_cls(**params)




def sample_transform(trial, cfg):

    # sample -> sometimes enabled sometimes not
    enabled = trial.suggest_categorical(
        "prep.transform.enabled",
        [True, False],
    )

    if not enabled:
        return {"enabled": False}

    name = trial.suggest_categorical(
        "prep.transform.name",
        cfg.names,
    )

    return {
        "enabled": True,
        "name": name,
    }




"""
The important thing you're learning here is that your pipeline doesn't need to know 
whether a transformation comes from NumPy, sklearn, 
PyTorch, or TensorFlow. You adapt it to the common fit() / transform() interface.
"""