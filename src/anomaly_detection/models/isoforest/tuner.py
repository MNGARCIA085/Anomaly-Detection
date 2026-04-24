

from anomaly_detection.models.base import BaseTuner
from anomaly_detection.models.isoforest.schemas import IsoForestConfig, IsoForestTuningConfig




class IsoForestTuner(BaseTuner):

    def sample_model_config(self, trial, tuning_cfg: IsoForestTuningConfig, runtime_params):

        n_estimators = trial.suggest_int(
            tuning_cfg.n_estimators.name,
            tuning_cfg.n_estimators.low,
            tuning_cfg.n_estimators.high
        )

        contamination = trial.suggest_float(
            tuning_cfg.contamination.name,
            tuning_cfg.contamination.low,
            tuning_cfg.contamination.high,
            log=tuning_cfg.contamination.log
        )

        return IsoForestConfig(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=42
        )


    def sample_training_config(self, trial, base_cfg):
        return None





#-------------------------------------------------
class IsoForestTunerv0(BaseTuner):

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
        return None



