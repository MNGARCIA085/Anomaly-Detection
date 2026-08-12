from sklearn.impute import SimpleImputer


IMPUTATION_REGISTRY = {
    "simple": SimpleImputer,
}


def create_imputer(name, **params):
    try:
        imputer_cls = IMPUTATION_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown imputation method: {name}. "
            f"Available: {list(IMPUTATION_REGISTRY)}"
        )

    return imputer_cls(**params)



def sample_imputation(trial, cfg):
    """Sample imputation for tuning."""

    enabled = trial.suggest_categorical(
        "prep.imputation.enabled",
        [True, False],
    )

    if not enabled:
        return {
            "enabled": False,
        }

    name = trial.suggest_categorical(
        "prep.imputation.name",
        cfg.names,
    )

    strategy = trial.suggest_categorical(
        "prep.imputation.strategy",
        cfg.strategy.choices,
    )

    return {
        "enabled": True,
        "name": name,
        "params": {
            "strategy": strategy,
        },
    }