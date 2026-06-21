from abc import ABC, abstractmethod

# Tuner
class BaseTuner(ABC):
    @abstractmethod
    def sample_model_config(self, trial, runtime_params, model_tuning_cfg): 
        pass
    
    # see if isoforest.. really needs this!!!
    @abstractmethod
    def sample_training_config(self, trial, training_tuning_cfg): 
        pass



"""
from abc import ABC, abstractmethod

class BaseTuner(ABC):
    @abstractmethod
    def sample_model_config(self, trial, runtime_params, model_tuning_cfg): 
        Sample architectural/hyperparameters (works for all models).
        pass

class DeepLearningTuner(BaseTuner, ABC):
    @abstractmethod
    def sample_training_config(self, trial, training_tuning_cfg): 
        Sample iterative optimization parameters (epochs, lr, batch size).
        pass
"""