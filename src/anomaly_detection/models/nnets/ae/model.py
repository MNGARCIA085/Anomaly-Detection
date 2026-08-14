import numpy as np

import torch
import torch.nn as nn

# architecture
from .schemas import AEConfig


from ...base_model import AnomalyWrapper

from ...persistence.torch import save_torch_model, load_torch_model






class AE(nn.Module):

    def __init__(self, cfg: AEConfig):
        super().__init__()
        self.config = cfg

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







# wrapper -> model + trainer
class AEWrapper(AnomalyWrapper):

    def __init__(
        self,
        model,
        trainer
    ):
        self.model = model
        self.trainer = trainer

    def fit(
        self,
        X_train,
        X_val=None
    ):

        self.model = self.trainer.fit(
            self.model,
            X_train,
            X_val
        )

        return self

    def get_scores(self, X):

        X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.model.eval()

        with torch.no_grad():

            recon = self.model(X)

            return torch.mean(
                (X - recon) ** 2,
                dim=1
            ).numpy()



    # binary preds
    def predict(self, X, threshold):

        scores = self.get_scores(X)

        return (
            scores > threshold
        ).astype(int)





    # property for input dim
    @property
    def input_dim(self):
        return self.model.config.input_dim


    # training history
    @property
    def history(self):
        return self.trainer.history


    # save and load
    def save(self, path):

        save_torch_model(
            self.model,
            path
        )

    @classmethod
    def load(cls, path):

        model = load_torch_model(
            AE,
            path
        )

        return cls(
            model=model,
            trainer=None
        )


