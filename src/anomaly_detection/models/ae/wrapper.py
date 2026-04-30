import torch
from anomaly_detection.models.base import AnomalyModel
from anomaly_detection.models.ae.tuner import AETuner
from anomaly_detection.models.ae.architecture import build_model
from anomaly_detection.models.ae.trainer import AETrainer


# wrapper class
class AutoencoderModel(AnomalyModel):
    
    def __init__(self, model, trainer):
        self.model = model
        self.trainer = trainer # injected, Only NNs see the Trainer


    def fit(self, X_train_prep, X_val_prep=None):
        self.trainer.train(self.model, X_train_prep, X_val_prep)


    def get_scores(self, X):
        X_t = torch.tensor(X, dtype=torch.float32)
        recon = self.model(X_t).detach()
        error = ((X_t - recon) ** 2).mean(dim=1).numpy()
        return error


# builder (separate later maybe)
def build_wrapper(model_cfg, training_cfg, runtime_params, trial=None, cfg=None):
    if trial is not None:
        tuner = AETuner()
        model_cfg = tuner.sample_model_config(trial, cfg.model_space, runtime_params)
        training_cfg = tuner.sample_training_config(trial, cfg.training_space)

    model = build_model(model_cfg, runtime_params)
    trainer = AETrainer(training_cfg)
    return AutoencoderModel(model, trainer)


# to think: https://chatgpt.com/c/69eaa917-7050-83e9-a65f-ec92e2e25fc8



