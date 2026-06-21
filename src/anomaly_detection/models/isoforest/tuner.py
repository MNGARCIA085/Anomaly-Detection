from anomaly_detection.tuning.base_tuner import BaseTuner
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







