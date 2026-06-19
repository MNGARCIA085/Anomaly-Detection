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