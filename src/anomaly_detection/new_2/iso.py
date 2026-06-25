import numpy as np
import optuna

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest



from anomaly_detection.new_2.preprocessing import PreprocessingPipeline # new


class IsoWrapper:

    def __init__(self, model):
        self.model = model

    def fit(
        self,
        X,
        y=None
    ):
        self.model.fit(X)
        return self

    def get_scores(self, X):

        return -self.model.decision_function(X)



#-------------entry


from anomaly_detection.new_2.registry import register

@register("iso")
class IsoEntry:

    @staticmethod
    def sample(trial):

        return {

            "prep": {
                "scaler": trial.suggest_categorical(
                    "scaler",
                    ["standard", "minmax"]
                )
            },

            "model": {

                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    50,
                    300
                ),

                "contamination": trial.suggest_float(
                    "contamination",
                    0.01,
                    0.2
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

        return PreprocessingPipeline(
            steps
        )

    @staticmethod
    def build(
        cfg,
        input_dim
    ):

        # my model, not need to define it custom like AE
        model = IsolationForest(
            n_estimators=cfg["model"]["n_estimators"],
            contamination=cfg["model"]["contamination"]
        )

        return IsoWrapper(
            model
        )