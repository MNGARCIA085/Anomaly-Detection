from abc import ABC, abstractmethod


class ThresholdStrategy(ABC):

    @abstractmethod
    def fit(self, scores):
        pass

    @abstractmethod
    def get_threshold(self):
        pass



#--------first strategy-----------------#
import numpy as np


# use with x_train, good if i train mostly on normal data
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
                "Threshold strategy has not been fitted."
            )

        return self.threshold



#-------------registry-------------#
#from .thresholds.quantile import QuantileThreshold


THRESHOLD_REGISTRY = {
    "quantile": QuantileThreshold,
}


def create_threshold_strategy(
    name,
    **params,
):
    strategy_cls = THRESHOLD_REGISTRY[name]

    return strategy_cls(**params)




"""
threshold:
  name: quantile
  params:
    quantile: 0.99

threshold: null
"""