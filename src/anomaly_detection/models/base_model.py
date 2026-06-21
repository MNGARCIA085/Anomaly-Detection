from abc import ABC, abstractmethod



"""
Both classes define the global contract for how models interact with your core system:

- AnomalyModel defines what a model does (fit, get_scores).

- AnomalyModelBuilder defines how an experiment instantiates a model dynamically.

Because they are both abstract foundational blueprints that don't contain any 
model-specific logic (like PyTorch or Sklearn code), 
keeping them in src/models/base_model.py is highly cohesive

"""

# Wrapper
class AnomalyModel(ABC):

    @abstractmethod
    def fit(self, X_train, X_val=None): 
        pass

    @abstractmethod
    def get_scores(self, X): 
        """ return scores """
        pass



