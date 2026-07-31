import numpy as np

import torch
import torch.nn as nn

# architecture
from .schemas import AEConfig


from ...base_model import AnomalyWrapper



from pathlib import Path
import joblib




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


    # property for input dim
    @property
    def input_dim(self):
        return self.model.config.input_dim


    """
    # save model
    def save(self, path):

        path = Path(path)
        path.mkdir(
            parents=True,
            exist_ok=True
        )

        # Save model weights
        torch.save(
            self.model.state_dict(),
            path / "weights.pt"
        )

        # Save everything needed to rebuild the architecture
        joblib.dump(
            self.model.config,
            path / "config.pkl"
        )

    # load model
    @classmethod
    def load(cls, path):

        path = Path(path)

        # Load configuration
        cfg = joblib.load(
            path / "config.pkl"
        )

        # Rebuild model
        model = AE(cfg)

        # Load weights
        model.load_state_dict(
            torch.load(
                path / "weights.pt",
                map_location="cpu"
            )
        )

        model.eval()

        # No trainer needed for inference
        return cls(
            model=model,
            trainer=None
        )
    """











#---------move later
# persistence/torch.py

def save_torch_model(model, path):

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    torch.save(
        model.state_dict(),
        path / "weights.pt"
    )

    joblib.dump(
        model.config,
        path / "config.pkl"
    )

def load_torch_model(model_cls, path):

    path = Path(path)

    cfg = joblib.load(
        path / "config.pkl"
    )

    model = model_cls(cfg)

    model.load_state_dict(
        torch.load(
            path / "weights.pt",
            map_location="cpu"
        )
    )

    model.eval()

    return model




"""
@classmethod
def load(cls, path):

    cfg = joblib.load(
        path / "config.pkl"
    )

    model = AE(cfg)

    model.load_state_dict(
        torch.load(path / "weights.pt")
    )

    model.eval()

    return cls(
        model=model,
        trainer=None
    )
"""






# too many duplicate code in save/load
# maybe add an abstraction
# or simple functions and use them here
# all torch models are saved and load probably in the same way 
    