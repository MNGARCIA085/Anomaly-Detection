from abc import ABC, abstractmethod


#--------Abstract class for entries (if needed move to BaseEntry)-----------#
class BaseModelEntry(ABC):
    """
    Contract for all anomaly detection model entries.
    Any class inheriting from this MUST implement these three methods.
    """


    @abstractmethod
    def sample(self, trial, tun_cfg):
        """Define the Optuna search space for this model."""
        pass


    @abstractmethod
    def build_preprocessor(self, prep_cfg):
        """Build and return the preprocessing pipeline."""
        pass


    @abstractmethod
    def adapt_input(self, X):
        pass


    @abstractmethod
    def build(self, model_cfg, training_cfg=None, input_dim=None):
        """
        Build and return the model wrapper.
        Accepts model_cfg as a required argument, and optional
        parameters: `training_cfg` and `input_dim`.
        """
        pass

    @abstractmethod
    def load(self, path):
        pass