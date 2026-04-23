from dataclasses import dataclass
from sklearn.ensemble import IsolationForest
from anomaly_detection.models.base import AnomalyModel



@dataclass
class IsoForestConfig:
    n_estimators=100 # not hardcode later; pass them all in config (hydra, but lets use this defaults)
    contamination=0.01
    random_state=42


def build_forest(cfg: IsoForestConfig):
    return IsolationForest(
            n_estimators=cfg.n_estimators,
            contamination=cfg.contamination,
            random_state=cfg.random_state # later -> more global
    )






def build_forest_tuning(cfg: IsoForestConfig, trial=None):

    n_estimators = cfg.n_estimators
    contamination = cfg.contamination

    if trial is not None:
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        contamination = trial.suggest_float("contamination", 0.001, 0.1, log=True)

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=cfg.random_state
    )

    return IsolationForestModel(model)




# wrapper
class IsolationForestModel(AnomalyModel):

    def __init__(self, model, trainer=None):
        self.model = model
        self.trainer = trainer # take away

    def fit(self, X_train, X_val): # manage X_val properly!!!!!!!
        self.model.fit(X_train)

    def get_scores(self, X):
        scores = self.model.decision_function(X)  # higher = more normal
        return -scores  # higher = more anomalous