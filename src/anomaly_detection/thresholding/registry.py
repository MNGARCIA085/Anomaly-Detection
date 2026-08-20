from .strategies import QuantileThreshold

THRESHOLD_REGISTRY = {
    "quantile": QuantileThreshold,
}


def create_threshold_strategy(
    name,
    **params,
):
    strategy_cls = THRESHOLD_REGISTRY[name]

    return strategy_cls(**params)