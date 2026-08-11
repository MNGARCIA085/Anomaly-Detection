from sklearn.decomposition import PCA, TruncatedSVD


DIMENSIONALITY_REDUCTION_REGISTRY = {
    "pca": PCA,
    "truncated_svd": TruncatedSVD,
}



def create_dimensionality_reducer(cfg):
    if not cfg["enabled"]:
        return None

    name = cfg["name"]

    try:
        reducer_cls = DIMENSIONALITY_REDUCTION_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown dimensionality reduction method: {name}. "
            f"Available: {list(DIMENSIONALITY_REDUCTION_REGISTRY)}"
        )

    return reducer_cls(**cfg.get("params", {}))





def sample_dimensionality_reducer(trial, cfg):


    print("\n", cfg)

    if not cfg.enabled:
        return {
            "enabled": False,
            "name": None,
            "params": {},
        }

    name = trial.suggest_categorical(
        "prep.dimensionality.name",
        cfg.names,
    )

    params = {}

    if name == "pca":
        params["n_components"] = trial.suggest_int(
            "prep.dimensionality.n_components",
            cfg.n_components.low,
            cfg.n_components.high,
        )

    return {
        "enabled": True,
        "name": name,
        "params": params,
    }