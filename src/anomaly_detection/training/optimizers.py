import torch


OPTIMIZER_REGISTRY = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}




def create_optimizer(cfg, parameters):
    optimizer_cls = OPTIMIZER_REGISTRY[cfg["name"]]

    return optimizer_cls(
        parameters,
        **cfg.get("params", {})
    )



"""
def create_optimizer(cfg, parameters):

    optimizer_cls = OPTIMIZER_REGISTRY[cfg.name]

    return optimizer_cls(
        parameters,
        **cfg.params
    )
"""