from abc import ABC, abstractmethod
import torch


# for now constructed from config


class BaseOptimizer(ABC):

    @staticmethod
    @abstractmethod
    def sample(trial, cfg):
        """Return optimizer config dictionary."""
        pass

    @staticmethod
    @abstractmethod
    def create(parameters, cfg):
        """Return instantiated torch optimizer."""
        pass





class AdamOptimizer(BaseOptimizer):

    @staticmethod
    def sample(trial, cfg):

        return {
            "name": "adam",
            "params": {
                "lr": trial.suggest_float(
                    "optimizer.lr",
                    cfg.lr.low,
                    cfg.lr.high,
                    log=cfg.lr.log,
                ),
                "betas": trial.suggest_categorical(
                    "optimizer.betas",
                    cfg.betas.choices,
                ),
            },
        }

    @staticmethod
    def create(parameters, cfg):

        return torch.optim.Adam(
            parameters,
            **cfg["params"],
        )






class SGDOptimizer(BaseOptimizer):

    @staticmethod
    def sample(trial, cfg):

        return {
            "name": "sgd",
            "params": {
                "lr": trial.suggest_float(
                    "optimizer.lr",
                    cfg.lr.low,
                    cfg.lr.high,
                    log=cfg.lr.log,
                ),
                "momentum": trial.suggest_float(
                    "optimizer.momentum",
                    cfg.momentum.low,
                    cfg.momentum.high,
                ),
                "weight_decay": trial.suggest_float(
                    "optimizer.weight_decay",
                    cfg.weight_decay.low,
                    cfg.weight_decay.high,
                    log=cfg.weight_decay.log,
                ),
            },
        }

    @staticmethod
    def create(parameters, cfg):

        return torch.optim.SGD(
            parameters,
            **cfg["params"],
        )



OPTIMIZER_REGISTRY = {
    "adam": AdamOptimizer,
    "sgd": SGDOptimizer,
}


def create_optimizer(cfg, parameters):

    optimizer_cls = OPTIMIZER_REGISTRY[cfg["name"]]

    return optimizer_cls.create(
        parameters,
        cfg,
    )




#-------sample optimizer------------#

def sample_optimizer(trial, cfg):

    optimizer_name = trial.suggest_categorical(
        "optimizer.name",
        cfg.names,
    )

    return OPTIMIZER_REGISTRY[
        optimizer_name
    ].sample(
        trial,
        cfg[optimizer_name],
    )





"""
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


"""
def create_optimizer(cfg, parameters):

    optimizer_cls = OPTIMIZER_REGISTRY[cfg.name]

    return optimizer_cls(
        parameters,
        **cfg.params
    )
"""