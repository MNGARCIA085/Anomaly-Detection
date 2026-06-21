

from anomaly_detection.tuning.base_tuner import BaseTuner
from anomaly_detection.models.ae.schemas import AEConfig, AETrainingConfig, AETrainingTuningConfig, AETuningConfig
from anomaly_detection.models.ae.trainer import PrintLossCallback, EarlyStopping



class AETuner(BaseTuner):

    def sample_model_config(self, trial, tuning_cfg: AETuningConfig, runtime_params):
        input_dim = runtime_params["input_dim"]

        n_layers = trial.suggest_int(
            tuning_cfg.n_layers.name,
            tuning_cfg.n_layers.low,
            tuning_cfg.n_layers.high
        )

        encoder_dims = [
            trial.suggest_int(
                f"{tuning_cfg.encoder_dim.name}_{i}",
                tuning_cfg.encoder_dim.low,
                tuning_cfg.encoder_dim.high
            )
            for i in range(n_layers)
        ]

        return AEConfig(
            input_dim=input_dim,
            encoder_dims=encoder_dims
        )



    def sample_training_config(self, trial, tuning_cfg: AETrainingTuningConfig):

        return AETrainingConfig(
            lr=trial.suggest_float(
                tuning_cfg.lr.name,
                tuning_cfg.lr.low,
                tuning_cfg.lr.high,
                log=tuning_cfg.lr.log
            ),
            batch_size=trial.suggest_categorical(
                tuning_cfg.batch_size.name,
                tuning_cfg.batch_size.choices
            ),
            epochs=tuning_cfg.epochs,

            # callbvakcs tests
            callbacks=[PrintLossCallback(), EarlyStopping(patience=3)]


        )



