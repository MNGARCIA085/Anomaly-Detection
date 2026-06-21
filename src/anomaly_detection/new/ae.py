import numpy as np
import optuna

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest


from anomaly_detection.new.registry import register


from anomaly_detection.new.preprocessing import PreprocessingPipeline



# architecture
class AE(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.enc = nn.Linear(input_dim, hidden_dim)
        self.dec = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):

        z = torch.relu(self.enc(x))
        return self.dec(z)







# trainer

class EarlyStopping:

    def __init__(self, patience=3, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta

        self.best = float("inf")
        self.counter = 0
        self.should_stop = False

    def __call__(self, val_loss):

        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True

class AETrainer:

    def __init__(
        self,
        lr,
        epochs,
        batch_size,
        callback=None
    ):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.callback = callback

    def fit(
        self,
        model,
        X_train,
        X_val=None
    ):

        model.train()

        opt = optim.Adam(
            model.parameters(),
            lr=self.lr
        )

        loss_fn = nn.MSELoss()

        X_train = torch.tensor(
            X_train,
            dtype=torch.float32
        )

        if X_val is not None:
            X_val = torch.tensor(
                X_val,
                dtype=torch.float32
            )

        for epoch in range(self.epochs):

            perm = torch.randperm(
                X_train.size(0)
            )

            X_train = X_train[perm]

            for i in range(
                0,
                X_train.size(0),
                self.batch_size
            ):

                batch = X_train[
                    i:i+self.batch_size
                ]

                loss = loss_fn(
                    model(batch),
                    batch
                )

                opt.zero_grad()
                loss.backward()
                opt.step()

            if (
                X_val is not None
                and self.callback is not None
            ):

                model.eval()

                with torch.no_grad():

                    recon = model(X_val)

                    val_loss = torch.mean(
                        (X_val - recon) ** 2
                    ).item()

                model.train()

                self.callback(val_loss)

                if self.callback.should_stop:
                    break

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
                "hidden_dim": trial.suggest_int(
                    "hidden_dim",
                    8,
                    64
                )
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

        model = AE(
            input_dim=input_dim,
            hidden_dim=cfg["model"]["hidden_dim"]
        )

        trainer = AETrainer(
            lr=cfg["training"]["lr"],
            epochs=cfg["training"]["epochs"],
            batch_size=cfg["training"]["batch_size"],
            callback=EarlyStopping(
                patience=3
            )
        )

        return AEWrapper(
            model,
            trainer
        )