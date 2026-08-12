from sklearn.feature_selection import VarianceThreshold


FEATURE_SELECTION_REGISTRY = {
    "variance_threshold": VarianceThreshold,
}


def create_feature_selector(name, **params):
    try:
        selector_cls = FEATURE_SELECTION_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown feature selection method: {name}. "
            f"Available: {list(FEATURE_SELECTION_REGISTRY)}"
        )

    return selector_cls(**params)



def sample_feature_selection(trial, cfg):

    if not cfg.enabled:
        return {
            "enabled": False,
        }

    name = trial.suggest_categorical(
        "prep.feature_selection.name",
        cfg.names,
    )

    threshold = trial.suggest_float(
        "prep.feature_selection.threshold",
        cfg.threshold.low,
        cfg.threshold.high,
    )

    return {
        "enabled": True,
        "name": name,
        "params": {
            "threshold": threshold,
        },
    }