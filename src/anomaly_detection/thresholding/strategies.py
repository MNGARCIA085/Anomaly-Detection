import numpy as np
from .base import ThresholdStrategy



# Quantile thresholding assumes that the training score distribution is predominantly 
# normal

class QuantileThreshold(ThresholdStrategy):

    def __init__(self, quantile=0.99):
        self.quantile = quantile
        self.threshold = None

    """
    def fit(self, scores):

        self.threshold = np.quantile(
            scores,
            self.quantile,
        )

        return self
    """


    def fit(
        self,
        train_scores,
        val_scores=None,
        y_val=None,
    ):
        self.threshold = np.quantile(
            train_scores,
            self.quantile,
        )

        return self


    def get_threshold(self):

        if self.threshold is None:
            raise RuntimeError(
                "Threshold has not been fitted."
            )

        return self.threshold




#-------------cons. sup. threshold-----------------#
import numpy as np
from sklearn.metrics import f1_score, recall_score


class ConstrainedF1Threshold(ThresholdStrategy):

    def __init__(
        self,
        min_recall=0.80,
    ):
        self.min_recall = min_recall
        self.threshold = None

    def fit(
        self,
        train_scores,
        val_scores=None,
        y_val=None,
    ):

        if val_scores is None or y_val is None:
            raise ValueError(
                "ConstrainedF1Threshold requires "
                "validation scores and labels."
            )

        if len(val_scores) != len(y_val):
            raise ValueError(
                "val_scores and y_val must have "
                "the same length."
            )

        thresholds = np.unique(val_scores)

        best_threshold = None
        best_f1 = -np.inf

        for threshold in thresholds:

            predictions = (
                val_scores >= threshold
            )

            recall = recall_score(
                y_val,
                predictions,
                zero_division=0,
            )

            if recall < self.min_recall:
                continue

            f1 = f1_score(
                y_val,
                predictions,
                zero_division=0,
            )

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        if best_threshold is None:
            raise ValueError(
                "No threshold satisfies the "
                f"minimum recall constraint: "
                f"{self.min_recall}"
            )

        self.threshold = best_threshold

        return self

    def get_threshold(self):

        if self.threshold is None:
            raise RuntimeError(
                "Threshold has not been fitted."
            )

        return self.threshold