import torch
import torch.nn as nn


from .schemas import TransformerAEConfig





import math
import torch
import torch.nn as nn

from anomaly_detection.models.base_model import AnomalyWrapper
from anomaly_detection.models.persistence.torch import (
    save_torch_model,
    load_torch_model,
)




# ============================================================
# Positional Encoding
# ============================================================

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, seq_len):

        super().__init__()

        position = torch.arange(
            seq_len,
            dtype=torch.float32
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float32
            )
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(
            seq_len,
            d_model
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        # (1, seq_len, d_model)
        self.register_buffer(
            "pe",
            pe.unsqueeze(0)
        )

    def forward(self, x):

        return x + self.pe[:, :x.size(1)]


# ============================================================
# Transformer Autoencoder
# ============================================================

class TransformerAE(nn.Module):

    def __init__(self, cfg: TransformerAEConfig):

        super().__init__()

        self.config = cfg

        # ----- Input projection -----

        self.input_projection = nn.Linear(
            cfg.input_dim,
            cfg.d_model
        )

        # ----- Positional encoding -----

        self.positional_encoding = PositionalEncoding(
            cfg.d_model,
            cfg.seq_len
        )

        # ----- Transformer encoder -----

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="relu",
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.num_encoder_layers,
        )

        # ----- Output projection -----

        self.output_projection = nn.Linear(
            cfg.d_model,
            cfg.input_dim
        )

    def forward(self, x):

        # x:
        # (batch, seq_len, input_dim)

        x = self.input_projection(x)

        x = self.positional_encoding(x)

        x = self.encoder(x)

        x = self.output_projection(x)

        return x


# ============================================================
# Wrapper
# ============================================================

class TransformerAEWrapper(AnomalyWrapper):

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

    # ----- Anomaly scores -----

    def get_scores(self, X):

        X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.model.eval()

        with torch.no_grad():

            recon = self.model(X)

            # Mean reconstruction error
            # over sequence and feature dimensions.
            return torch.mean(
                (X - recon) ** 2,
                dim=(1, 2)
            ).numpy()

    # ----- Binary predictions -----

    def predict(self, X, threshold):

        scores = self.get_scores(X)

        return (
            scores > threshold
        ).astype(int)

    # ----- Input dimension -----

    @property
    def input_dim(self):

        return self.model.config.input_dim

    # ----- Training history -----

    @property
    def history(self):

        return self.trainer.history

    # ----- Save / Load -----

    def save(self, path):

        save_torch_model(
            self.model,
            path
        )

    @classmethod
    def load(cls, path):

        model = load_torch_model(
            TransformerAE,
            path
        )

        return cls(
            model=model,
            trainer=None
        )