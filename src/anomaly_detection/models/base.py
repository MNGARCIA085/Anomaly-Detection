from abc import ABC, abstractmethod


# later -> move ti a core fodler!!!


# Wrapper
class AnomalyModel(ABC):

    @abstractmethod
    def fit(self, X_train, X_val=None): 
        pass

    @abstractmethod
    def get_scores(self, X): 
        """ return scores """
        pass


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
core/
  wrapper.py      ← AnomalyModel
  tuner.py        ← BaseTuner
  builder.py      ← AnomalyModelBuilder
  experiment.py

models/
  ae/
    builder.py
    tuner.py      ← AETuner
    model.py
    trainer.py
    config.py
"""