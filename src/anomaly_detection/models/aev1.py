from dataclasses import dataclass
from anomaly_detection.models.base import AnomalyModel

import torch
import torch.nn as nn


from typing import List


# dataclass for Model
@dataclass
class AEConfig:
    input_dim: int
    encoder_dims: List[int] = (8, 4)
    decoder_dims: List[int] = (8,)  # last layer (output) will use input_dim


@dataclass
class AETrainingConfig:
    lr: float
    batch_size: int
    epochs: int


# Model
class AE(nn.Module):

    def __init__(self, cfg: AEConfig):
        super().__init__()

        # ----- Encoder -----
        encoder_layers = []
        in_dim = cfg.input_dim
        for dim in cfg.encoder_dims:
            encoder_layers.append(nn.Linear(in_dim, dim))
            encoder_layers.append(nn.ReLU())
            in_dim = dim
        encoder_layers = encoder_layers[:-1]  # remove last ReLU if you want

        self.encoder = nn.Sequential(*encoder_layers)

        # ----- Decoder -----
        decoder_layers = []
        in_dim = cfg.encoder_dims[-1]
        for dim in cfg.decoder_dims:
            decoder_layers.append(nn.Linear(in_dim, dim))
            decoder_layers.append(nn.ReLU())
            in_dim = dim

        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))




# wrapper
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



##########



def build_ae_model(cfg, runtime_params, trial=None):

    input_dim = runtime_params["input_dim"]

    encoder_dims = list(cfg.encoder_dims or [64, 32])

    if trial is not None:
        n_layers = trial.suggest_int("n_layers", 1, 3)

        encoder_dims = [
            trial.suggest_int(f"enc_dim_{i}", 16, 128)
            for i in range(n_layers)
        ]

    ae_cfg = AEConfig(
        input_dim=input_dim,
        encoder_dims=encoder_dims,
        decoder_dims=encoder_dims[:-1][::-1]
    )

    return AE(ae_cfg)


def build_ae_wrapper(model_cfg, training_cfg, runtime_params, trial=None):

    input_dim = runtime_params["input_dim"]

    # ----- Defaults from config -----
    encoder_dims = list(model_cfg.encoder_dims)

    lr = training_cfg.lr
    batch_size = training_cfg.batch_size
    epochs = training_cfg.epochs

    # ----- 🔥 Optuna overrides -----
    if trial is not None:
        n_layers = trial.suggest_int("n_layers", 1, 3)

        encoder_dims = [
            trial.suggest_int(f"enc_dim_{i}", 16, 128)
            for i in range(n_layers)
        ]

        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    # ----- Config -----
    ae_cfg = AEConfig(
        input_dim=input_dim,
        encoder_dims=encoder_dims,
        # keep decoder implicit OR mirror it explicitly:
        decoder_dims=encoder_dims[:-1][::-1]
    )

    # ----- Build -----
    model = AE(ae_cfg)

    trainer = AETrainer()
    #    lr=lr,
    #    batch_size=batch_size,
    #    epochs=epochs
    #)

    return AutoencoderModel(model, trainer)





# Trainer
class AETrainer():
	# for now very simpel, later, also rain with x_val
	def train(self, model, X_train, X_val): # pass model here, not need to save state
		optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
		criterion = nn.MSELoss()

		# ---- Train (only normal data) ----
		X_t = torch.tensor(X_train, dtype=torch.float32)

		for epoch in range(50): # pass it later to training config!!!!!
		    optimizer.zero_grad()
		    recon = model(X_t)
		    loss = criterion(recon, X_t)
		    loss.backward()
		    optimizer.step()

		return model



