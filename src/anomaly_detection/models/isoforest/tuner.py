

from anomaly_detection.models.base import BaseTuner
from anomaly_detection.models.isoforest.schemas import IsoForestConfig 



class IsoForestTuner(BaseTuner):

    # for tng, with oiptuna; returns model config!!!
    def sample_model_config(self, trial, base_cfg, runtime_params): # see later conf.
        
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        contamination = trial.suggest_float("contamination", 0.001, 0.1, log=True)

        return IsoForestConfig(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=42
        )


    # not hardcode suggetsions later, pass them from a config (tuning_cfg)
    def sample_training_config(self, trial, base_cfg):
        pass



