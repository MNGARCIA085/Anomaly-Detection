from anomaly_detection.models.base import AnomalyModel
from anomaly_detection.models.isoforest.tuner import IsoForestTuner

from anomaly_detection.models.isoforest.architecture import build_model

# Model wrapper
class IsolationForestModel(AnomalyModel):

    def __init__(self, model): # not need for a tariner
        self.model = model

    def fit(self, X_train, X_val=None): # see later the need for X_val
        self.model.fit(X_train)

    def get_scores(self, X):
        scores = self.model.decision_function(X)  # higher = more normal
        return -scores  # higher = more anomalous



# builder
def build_wrapper(model_cfg, runtime_params, trial=None): # training_config?
    if trial is not None:
        tuner = IsoForestTuner()
        model_cfg = tuner.sample_model_config(trial, model_cfg, runtime_params)

    print('cfg', model_cfg)

    model = build_model(model_cfg) # runtime_params?
    return IsolationForestModel(model)



"""
# builder
def build_wrapper(model_cfg, training_cfg, runtime_params, trial=None):
    if trial is not None:
        tuner = AETuner()
        model_cfg = tuner.sample_model_config(trial, model_cfg, runtime_params)
        training_cfg = tuner.sample_training_config(trial, training_cfg)

    model = build_model(model_cfg, runtime_params)
    trainer = AETrainer(training_cfg)
    return AutoencoderModel(model, trainer)
"""