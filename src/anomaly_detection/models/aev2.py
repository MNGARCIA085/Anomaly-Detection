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







##########


"""
build_ae_model
sample_ae_model_cfg
sample_ae_training_cfg
build_ae_wrapper
"""



# it only buidls the model
def build_ae_model(cfg, runtime_params):

    input_dim = runtime_params["input_dim"]

    ae_cfg = AEConfig(
        input_dim=input_dim,
        encoder_dims=cfg.encoder_dims,
        decoder_dims=cfg.encoder_dims[:-1][::-1]
    )

    return AE(ae_cfg)



# for tng, with oiptuna
def sample_ae_model_cfg(base_cfg, trial, runtime_params): # Add runtime_params here

    # Now this will work:
    input_dim = runtime_params["input_dim"]
    
    # Also, ensure 'n_layers' is defined. 
    # If it's not in base_cfg, you might want to suggest it via trial:
    # n_layers = trial.suggest_int("n_layers", 1, 3) 

    encoder_dims = [
        trial.suggest_int(f"enc_dim_{i}", 16, 128)
        for i in range(n_layers)
    ]

    return AEConfig(
        input_dim=input_dim, # Make sure to pass this to the dataclass
        encoder_dims=encoder_dims
    )



# to use with optuna
def sample_ae_training_cfg(base_cfg, trial):

    return AETrainingConfig(
        lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
        epochs=base_cfg.epochs  # usually fixed (don’t tune at first)
    )



# buidl ae wraper that composes
def build_ae_wrapper(model_cfg, training_cfg, runtime_params, trial=None):

    # --- sample configs ---
    if trial is not None:
        # Pass runtime_params here!
        model_cfg = sample_ae_model_cfg(model_cfg, trial, runtime_params) 
        training_cfg = sample_ae_training_cfg(training_cfg, trial)

    # --- build components ---
    model = build_ae_model(model_cfg, runtime_params)
    trainer = AETrainer(training_cfg)

    return AutoencoderModel(
        model,
        trainer
    )







# Trainer
class AETrainer():


    def __init__(self, cfg: AETrainingConfig):
        self.lr = cfg.lr  
        self.epochs = cfg.epochs
        self.batch_size = cfg.batch_size

    # later -> add dataloaders...


    # for now very simpel, later, also rain with x_val
    def train(self, model, X_train, X_val): # pass model here, not need to save state
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        # ---- Train (only normal data) ----
        X_t = torch.tensor(X_train, dtype=torch.float32)

        for epoch in range(self.epochs): # pass it later to training config!!!!!
            optimizer.zero_grad()
            recon = model(X_t)
            loss = criterion(recon, X_t)
            loss.backward()
            optimizer.step()

        return model



