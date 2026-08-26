from abc import ABC, abstractmethod




#----------Anomaly Wrapper------------------#
class AnomalyWrapper(ABC):
    """ Defines what a model does"""

    @abstractmethod
    def fit(self, X_train, X_val=None): 
        pass

    @abstractmethod
    def get_scores(self, X): 
        """ return scores """
        pass


    @abstractmethod
    def predict(self, X, threshold=None):
        """ returns binary preds"""
        pass

    # save the model
    @abstractmethod
    def save(self, path): 
        pass


    # property for history
    @property
    @abstractmethod
    def history(self):
        pass

