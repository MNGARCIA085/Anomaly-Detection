import numpy as np
from .base import ThresholdStrategy



# Quantile thresholding assumes that the training score distribution is predominantly 
# normal

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
