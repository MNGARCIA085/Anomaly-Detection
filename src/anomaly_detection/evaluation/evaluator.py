# Evaluator & ThresholdSelector
import numpy as np
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score


#---------------------THRESHOLD----------------------#
class ThresholdSelector:

    @staticmethod
    def from_pr_curve(y, scores):
        precision, recall, thresholds = precision_recall_curve(y, scores)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        return thresholds[np.argmax(f1)]



#---------------------EVALUATOR----------------------#
class Evaluator:

    def evaluate(self, y_true, scores, threshold):
        y_pred = scores > threshold
        return {
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
        }

