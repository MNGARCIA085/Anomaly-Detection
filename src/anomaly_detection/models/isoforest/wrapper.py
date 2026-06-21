from anomaly_detection.models.base_model import AnomalyModel
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





# pass later to schema:https://gemini.google.com/app/20b9f9d20930b3cd
# builder (new file????)
def build_wrapper(model_cfg, runtime_params, training_cfg=None, trial=None, tuning_cfg=None): # training_config?
    
    if trial is not None:
        tuner = IsoForestTuner()
        # improve later!!!!!!
        model_tuning_cfg = IsoForestTuningConfig(
            n_estimators=IntParam("n_estimators", 50, 300),
            contamination=FloatParam("contamination", 0.001, 0.1, log=True)
        )
        model_cfg = tuner.sample_model_config(trial, model_tuning_cfg, runtime_params)

    model = build_model(model_cfg) # runtime_params?
    return IsolationForestModel(model)


