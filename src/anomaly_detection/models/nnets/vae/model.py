import numpy as np

import torch
import torch.nn as nn

from .schemas import VAEConfig

from ...base_model import AnomalyWrapper

from ...persistence.torch import save_torch_model, load_torch_model






class VAE(nn.Module):

    def __init__(self, cfg: VAEConfig):
        super().__init__()

        self.config = cfg

        # -------- Encoder backbone --------

        encoder_layers = []

        in_dim = cfg.input_dim

        for dim in cfg.encoder_dims:
            encoder_layers.append(nn.Linear(in_dim, dim))
            encoder_layers.append(nn.ReLU())
            in_dim = dim

        encoder_layers = encoder_layers[:-1]

        self.encoder = nn.Sequential(*encoder_layers)

        # latent parameters

        self.fc_mu = nn.Linear(in_dim, cfg.latent_dim)
        self.fc_logvar = nn.Linear(in_dim, cfg.latent_dim)

        # -------- Decoder --------

        decoder_layers = []

        in_dim = cfg.latent_dim

        for dim in cfg.decoder_dims:
            decoder_layers.append(nn.Linear(in_dim, dim))
            decoder_layers.append(nn.ReLU())
            in_dim = dim

        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):

        h = self.encoder(x)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)

        eps = torch.randn_like(std)

        return mu + eps * std

    def decode(self, z):

        return self.decoder(z)

    def forward(self, x):

        mu, logvar = self.encode(x)

        z = self.reparameterize(mu, logvar)

        recon = self.decode(z)

        return recon, mu, logvar


class VAEWrapper(AnomalyWrapper):

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

            recon, _, _ = self.model(X)

            return torch.mean(
                (X - recon) ** 2,
                dim=1
            ).numpy()

    @property
    def input_dim(self):
        return self.model.config.input_dim

    @property
    def history(self):
        return self.trainer.history

    def save(self, path):

        save_torch_model(
            self.model,
            path
        )

    @classmethod
    def load(cls, path):

        model = load_torch_model(
            VAE,
            path
        )

        return cls(
            model=model,
            trainer=None
        )