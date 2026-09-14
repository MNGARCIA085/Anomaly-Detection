from .strategies import QuantileThreshold, ConstrainedF1Threshold

THRESHOLD_REGISTRY = {
    "quantile": QuantileThreshold,
     "constrained_f1": ConstrainedF1Threshold,
}


def create_threshold_strategy(name, **params):
    try:
        strategy_cls = THRESHOLD_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown threshold strategy: {name}")

    return strategy_cls(**params)


