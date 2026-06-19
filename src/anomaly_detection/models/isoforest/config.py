from anomaly_detection.models.isoforest.schemas import IsoForestConfig,IsoForestTuningConfig
from anomaly_detection.core.schemas import IntParam, FloatParam, CategoricalParam



#------------------------for training only------------------

# hardcoded for quick tests
def build_iso_training_config(cfg):
    return IsoForestConfig(
        n_estimators = 100,
        contamination = 0.01
    )



def build_iso_tuning_config(cfg):
    return IsoForestTuningConfig(
        n_estimators = 100,
        contamination = 0.01
    )


