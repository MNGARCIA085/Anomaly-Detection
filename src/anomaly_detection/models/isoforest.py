from dataclasses import dataclass
from sklearn.ensemble import IsolationForest
from anomaly_detection.models.base import AnomalyModel



@dataclass
class IsoForestConfig:
    n_estimators=100, # not hardcode later; pass them all in config (hydra, but lets use this defaults)
    contamination=0.01,
    random_state=42


def build_forest(cfg: IsoForestConfig):
    return IsolationForest(
            n_estimators=cfg.n_estimators,
            contamination=cfg.contamination,
            random_state=cfg.random_state # later -> more global
    )



# wrapper
class IsolationForestModel(AnomalyModel):

    def __init__(self, model, trainer):
        self.model = model
        self.trainer = trainer

    def fit(self, X_train, X_val): # manage X_val properly!!!!!!!
        self.model.fit(X_train)

    def get_scores(self, X):
        scores = self.model.decision_function(X)  # higher = more normal
        return -scores  # higher = more anomalous