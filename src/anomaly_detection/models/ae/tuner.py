

from anomaly_detection.models.base import BaseTuner
from anomaly_detection.models.ae.schemas import AEConfig, AETrainingConfig, AETrainingTuningConfig, AETuningConfig




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


        )









class AETunerv0(BaseTuner):

    # for tuning with optuna
    def sample_model_config(self, trial, base_cfg, runtime_params):
        
        input_dim = runtime_params["input_dim"]
        
        # Also, ensure 'n_layers' is defined. 
        # If it's not in base_cfg, you might want to suggest it via trial:
        
        n_layers = trial.suggest_int("n_layers", 1, 3) 

        encoder_dims = [
            trial.suggest_int(f"enc_dim_{i}", 16, 128)
            for i in range(n_layers)
        ]

        return AEConfig(
            input_dim=input_dim,
            encoder_dims=encoder_dims
        )



    # not hardcode suggetsions later, pass them from a config (tuning_cfg)
    def sample_training_config(self, trial, base_cfg):

        return AETrainingConfig(
            lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
            epochs=10 
        )
