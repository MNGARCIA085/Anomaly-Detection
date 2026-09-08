

from pathlib import Path
from types import SimpleNamespace

import joblib
import torch
from omegaconf import DictConfig, OmegaConf


def save_torch_model(model, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Save model weights
    torch.save(
        model.state_dict(),
        path / "weights.pt"
    )

    # Save model config
    cfg = model.config

    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(
            cfg,
            resolve=True
        )

    joblib.dump(
        cfg,
        path / "config.pkl"
    )


def load_torch_model(model_cls, path):
    path = Path(path)

    # Load config
    cfg = joblib.load(
        path / "config.pkl"
    )

    # Models currently expect attribute access
    # e.g. cfg.input_dim, cfg.encoder_dims
    if isinstance(cfg, dict):
        cfg = SimpleNamespace(**cfg)

    # Reconstruct model
    model = model_cls(cfg)

    # Load weights
    model.load_state_dict(
        torch.load(
            path / "weights.pt",
            map_location="cpu"
        )
    )

    model.eval()

    return model


"""
from pathlib import Path
import joblib
import torch

from types import SimpleNamespace # workaround for now!!!


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

    if isinstance(cfg, dict):
        cfg = SimpleNamespace(**cfg)

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