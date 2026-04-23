from .schemas import IsoForestConfig


# Model
# ... not really needed, ill use sklearns
from sklearn.ensemble import IsolationForest



def build_model(cfg: IsoForestConfig):
    return IsolationForest(
            n_estimators=cfg.n_estimators,
            contamination=cfg.contamination,
            random_state=cfg.random_state # later -> more global
    )