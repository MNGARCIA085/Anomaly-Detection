from dataclasses import dataclass
from anomaly_detection.models.schemas import IntParam, FloatParam, CategoricalParam


@dataclass
class IsoForestConfig:
    n_estimators: int = 100
    contamination: float = 0.01 # carefull with this, i will alredy tune theshold
    random_state: int = 42 # more global later



@dataclass
class IsoForestTuningConfig:
    n_estimators: IntParam
    contamination: FloatParam