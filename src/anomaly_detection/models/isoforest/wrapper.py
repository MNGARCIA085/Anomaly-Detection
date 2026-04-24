from anomaly_detection.models.base import AnomalyModel
from anomaly_detection.models.isoforest.tuner import IsoForestTuner
from anomaly_detection.models.isoforest.architecture import build_model
from anomaly_detection.models.schemas import IntParam, FloatParam
from .schemas import IsoForestTuningConfig




# Model wrapper
class IsolationForestModel(AnomalyModel):

    def __init__(self, model): # not need for a tariner
        self.model = model

    def fit(self, X_train, X_val=None): # see later the need for X_val
        self.model.fit(X_train)

    def get_scores(self, X):
        scores = self.model.decision_function(X)  # higher = more normal
        return -scores  # higher = more anomalous






# builder (new file????)
def build_wrapper(model_cfg, runtime_params, trial=None, cfg=None): # training_config?
    
    if trial is not None:
        tuner = IsoForestTuner()
        model_tuning_cfg = IsoForestTuningConfig(
            n_estimators=IntParam("n_estimators", 50, 300),
            contamination=FloatParam("contamination", 0.001, 0.1, log=True)
        )
        model_cfg = tuner.sample_model_config(trial, model_tuning_cfg, runtime_params)

    model = build_model(model_cfg) # runtime_params?
    return IsolationForestModel(model)


