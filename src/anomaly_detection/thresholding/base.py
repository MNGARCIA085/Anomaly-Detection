from abc import ABC, abstractmethod

class ThresholdStrategy(ABC):

    #@abstractmethod
    #def fit(self, scores):
    #    pass


    @abstractmethod
    def fit(
        self,
        train_scores=None,
        val_scores=None,
        y_val=None,
    ):
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
