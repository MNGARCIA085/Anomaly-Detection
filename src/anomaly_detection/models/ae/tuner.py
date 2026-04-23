

from anomaly_detection.models.base import BaseTuner
from anomaly_detection.models.ae.schemas import AEConfig, AETrainingConfig



class AETuner(BaseTuner):

    # for tng, with optuna; returns model config!!!
    def sample_model_config(self, trial, base_cfg, runtime_params): # see later conf.
        
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
