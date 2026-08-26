from .strategies import QuantileThreshold

THRESHOLD_REGISTRY = {
    "quantile": QuantileThreshold,
}


def create_threshold_strategy(name, **params):
    try:
        strategy_cls = THRESHOLD_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown threshold strategy: {name}")

    return strategy_cls(**params)



"""
def create_threshold_strategy(
    name,
    **params,
):
    strategy_cls = THRESHOLD_REGISTRY[name]

    return strategy_cls(**params)
"""