

"""
Wrapper
  get_scores(X)          → continuous scores
  predict(X, threshold)  → binary predictions

Thresholding
  fit(scores)            → calculates/stores threshold
  get_threshold()        → returns threshold
"""


#Quantile thresholding assumes that the training score distribution is predominantly norma

#-----------Base---------#
from abc import ABC, abstractmethod





class ThresholdStrategy(ABC):

    @abstractmethod
    def fit(self, scores):
        pass

    @abstractmethod
    def get_threshold(self):
        pass


#--------strategies---------#
import numpy as np




# prob. only appropiate if i train mostly with std data
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

    def get_threshold(self):

        if self.threshold is None:
            raise RuntimeError(
                "Threshold has not been fitted."
            )

        return self.threshold



# --------registry-------------
#from .thresholds.quantile import QuantileThreshold
#from .thresholds.native import NativeThreshold


THRESHOLD_REGISTRY = {
    "quantile": QuantileThreshold,
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

            self.strategy = (
                create_threshold_strategy(
                    config["name"],
                    **config.get("params", {}),
                )
            )

    def fit(self, scores):

        if self.strategy is not None:
            self.strategy.fit(scores)

        return self

    def get_threshold(self):

        if self.strategy is None:
            return None

        return self.strategy.get_threshold()

    def save(self, path):

        joblib.dump(
            self,
            path,
        )

    @classmethod
    def load(cls, path):

        return joblib.load(path)