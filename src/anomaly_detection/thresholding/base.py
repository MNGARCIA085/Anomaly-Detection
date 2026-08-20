from abc import ABC, abstractmethod

class ThresholdStrategy(ABC):

    @abstractmethod
    def fit(self, scores):
        pass

    @abstractmethod
    def get_threshold(self):
        pass






"""
Wrapper
  get_scores(X)          → continuous scores
  predict(X, threshold)  → binary predictions

Thresholding
  fit(scores)            → calculates/stores threshold
  get_threshold()        → returns threshold
"""
