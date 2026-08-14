

#-----------Base---------#
from abc import ABC, abstractmethod


class ThresholdStrategy(ABC):

    @abstractmethod
    def predict(
        self,
        scores,
    ):
        pass



#--------strategies---------#
import numpy as np

#from .base import ThresholdStrategy


class QuantileThreshold(ThresholdStrategy):

    def __init__(self, quantile=0.99):
        self.quantile = quantile
        self.threshold = None

    def fit(self, scores):

        self.threshold = np.quantile(
            scores,
            self.quantile,
        )

        return self

    def predict(self, scores):

        if self.threshold is None:
            raise RuntimeError(
                "Threshold has not been fitted."
            )

        return (
            scores > self.threshold
        ).astype(int)




class NativeThreshold(ThresholdStrategy):

    def __init__(self, threshold):
        self.threshold = threshold

    def fit(self, scores):
        return self

    def predict(self, scores):

        if self.threshold is None:
            raise RuntimeError(
                "Threshold has not been set."
            )

        return (
            scores > self.threshold
        ).astype(int)




# --------registry-------------
#from .thresholds.quantile import QuantileThreshold
#from .thresholds.native import NativeThreshold


THRESHOLD_REGISTRY = {
    "quantile": QuantileThreshold,
    "native": NativeThreshold,
}


def create_threshold_strategy(
    name,
    **params,
):
    strategy_cls = THRESHOLD_REGISTRY[name]

    return strategy_cls(**params)



#---------thresholding------------#
#from .threshold_registry import (
#    create_threshold_strategy,
#)


import joblib

class Thresholding:

    def __init__(self, config):
        self.config = config

        self.strategy = None

        if config:
            self.strategy = create_threshold_strategy(
                config["name"],
                **config.get("params", {}),
            )

    def fit(self, scores):
        if self.strategy is not None:
            self.strategy.fit(scores)

        return self

    def predict(self, scores):
        if self.strategy is None:
            return None

        return self.strategy.predict(scores)


    # save and load
    def save(self, path):

        joblib.dump(
            self,
            path,
        )

    @classmethod
    def load(cls, path):

        return joblib.load(path)