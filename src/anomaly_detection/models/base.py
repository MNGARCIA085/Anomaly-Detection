from abc import ABC, abstractmethod


class AnomalyModel(ABC):

    @abstractmethod
    def fit(self, X_train, X_val=None): 
        pass

    @abstractmethod
    def get_scores(self, X): 
        """ return scores """
        pass




