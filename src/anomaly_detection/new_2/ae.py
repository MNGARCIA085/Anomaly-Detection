import numpy as np
import optuna

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest


from anomaly_detection.new_2.registry import register


from anomaly_detection.new_2.preprocessing import PreprocessingPipeline



# architecture





from dataclasses import dataclass, field
from typing import List, Any
from anomaly_detection.models.schemas import IntParam, FloatParam, CategoricalParam


# dataclass for Model
@dataclass
class AEConfig:
    input_dim: int
    encoder_dims: List[int] = (8, 4)
    decoder_dims: List[int] = (8,)


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



# trainer
# callbacks interface
class Callback:
    def on_train_start(self, state): pass
    def on_epoch_start(self, state): pass
    def on_epoch_end(self, state): pass
    def on_train_end(self, state): pass


class PrintLossCallback(Callback):
    def on_epoch_end(self, state):
        print(f"Epoch {state.epoch} - Train Loss: {state.train_loss:.4f} - Val Loss: {state.val_loss:.4f}")


class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.best = float("inf")
        self.counter = 0

    def on_epoch_end(self, state):
        if state.val_loss is None:
            return

        if state.val_loss < self.best:
            self.best = state.val_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            print('ES triggreed')
            state.stop_training = True






# dataloader
import torch
from torch.utils.data import Dataset

class AEDataset(Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]




#-----------------------------TRAINER--------------------------#

from torch.utils.data import DataLoader
import torch.nn as nn
import torch



@dataclass
class AETrainingConfig:
    lr: float
    batch_size: int
    epochs: int
    device: str = "cpu"
    callbacks: List[Any] = field(default_factory=list)
    shuffle: bool = True
    num_workers: int = 0


@dataclass
class TrainState:
    epoch: int = 0
    train_loss: float = 0.0
    val_loss: float = None
    model: Any = None
    stop_training: bool = False





# -----------------------
# Trainer
# -----------------------
# later -> optim... as DI -> it seems to work well
class AETrainer:

    def __init__(self, cfg: AETrainingConfig):
        self.cfg = cfg
        self.callbacks = cfg.callbacks or []

    def _call_callbacks(self, hook, state):
        for cb in self.callbacks:
            getattr(cb, hook, lambda x: None)(state)

    def fit(self, model, X_train, X_val=None): # train
        device = self.cfg.device
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.lr)
        criterion = nn.MSELoss()

        # ---- DataLoaders ----
        train_loader = DataLoader(
            AEDataset(X_train),
            batch_size=self.cfg.batch_size,
            shuffle=self.cfg.shuffle,
            num_workers=self.cfg.num_workers
        )

        val_loader = None
        if X_val is not None:
            val_loader = DataLoader(
                AEDataset(X_val),
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=self.cfg.num_workers
            )

        state = TrainState(model=model)
        self._call_callbacks("on_train_start", state)

        for epoch in range(self.cfg.epochs):
            state.epoch = epoch
            epoch_loss = 0.0

            self._call_callbacks("on_epoch_start", state)

            model.train()
            for batch in train_loader:
                batch = batch.to(device)

                optimizer.zero_grad()
                recon = model(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch.size(0)

            epoch_loss /= len(train_loader.dataset)
            state.train_loss = epoch_loss

            # ---- Validation ----
            if val_loader is not None:
                model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(device)
                        recon = model(batch)
                        loss = criterion(recon, batch)
                        val_loss += loss.item() * batch.size(0)

                val_loss /= len(val_loader.dataset)
                state.val_loss = val_loss

                model.train()

            self._call_callbacks("on_epoch_end", state)

            if state.stop_training:
                break

        self._call_callbacks("on_train_end", state)

        return model





# wrapper -> model + trainer
class AEWrapper:

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



# entry
@register("ae")
class AEEntry:
    """
    Model entry responsible for assembling all Autoencoder-specific components.

    Acts as the integration point between the experiment framework and the AE
    implementation by encapsulating:

    - hyperparameter search space definition (`sample`)
    - preprocessing construction (`build_preprocessor`)
    - model and training assembly (`build`)

    This class allows the experiment/tuning pipeline to remain model-agnostic:
    callers interact with a common entry interface without knowing how the AE
    is configured internally.

    Responsibilities:
        - Define tunable preprocessing, model, and training parameters
        - Build the preprocessing pipeline for AE workflows
        - Construct and return a fully configured AE wrapper

    Does NOT:
        - execute training
        - run evaluation
        - perform logging
        - orchestrate experiments

    Expected interface:
        sample(trial) -> dict
        build_preprocessor(cfg) -> PreprocessingPipeline
        build(cfg, input_dim) -> ModelWrapper
    """

    @staticmethod
    def sample(trial):

        return {

            "prep": {
                "scaler": trial.suggest_categorical(
                    "scaler",
                    ["standard", "minmax"]
                ),

                "use_pca": trial.suggest_categorical(
                    "use_pca",
                    [True, False]
                ),

                "pca_dim": trial.suggest_int(
                    "pca_dim",
                    2,
                    10
                ),
            },

            "model": {

                "encoder_dims": [
                    trial.suggest_int("enc1", 16, 64),
                    trial.suggest_int("enc2", 4, 32),
                ],

                "decoder_dims": [
                    trial.suggest_int("dec1", 16, 64)
                ]
            },

            "training": {

                "lr": trial.suggest_float(
                    "lr",
                    1e-4,
                    1e-2,
                    log=True
                ),

                "epochs": trial.suggest_int(
                    "epochs",
                    5,
                    20
                ),

                "batch_size": trial.suggest_categorical(
                    "batch_size",
                    [32, 64]
                )
            }
        }

    @staticmethod
    def build_preprocessor(cfg):

        steps = []

        if cfg["prep"]["scaler"] == "standard":
            steps.append(StandardScaler())

        elif cfg["prep"]["scaler"] == "minmax":
            steps.append(MinMaxScaler())

        if cfg["prep"]["use_pca"]:

            steps.append(
                PCA(
                    n_components=cfg["prep"]["pca_dim"]
                )
            )

        return PreprocessingPipeline(
            steps
        )

    @staticmethod
    def build(
        cfg,
        input_dim
    ):

        model_cfg = AEConfig(
            input_dim=input_dim,
            encoder_dims=cfg["model"]["encoder_dims"],
            decoder_dims=cfg["model"]["decoder_dims"],
        )

        model = AE(model_cfg)


        trainer_cfg = AETrainingConfig(
                lr=cfg["training"]["lr"],
                epochs=cfg["training"]["epochs"],
                batch_size=cfg["training"]["batch_size"],
                callbacks=[
                    EarlyStopping(patience=3),
                    PrintLossCallback(),
                ]
            )

        trainer = AETrainer(trainer_cfg)

            

        return AEWrapper(
            model,
            trainer
        )