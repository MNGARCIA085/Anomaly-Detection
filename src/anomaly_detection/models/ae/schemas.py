from dataclasses import dataclass
from typing import List
from anomaly_detection.models.schemas import IntParam, FloatParam, CategoricalParam


# dataclass for Model
@dataclass
class AEConfig:
    input_dim: int
    encoder_dims: List[int] = (8, 4)
    decoder_dims: List[int] = (8,)


@dataclass
class AETrainingConfig:
    lr: float
    batch_size: int
    epochs: int


@dataclass
class AETuningConfig:
    n_layers: IntParam
    encoder_dim: IntParam


@dataclass
class AETrainingTuningConfig:
    lr: FloatParam
    batch_size: CategoricalParam
    epochs: int  # fixed, not tuned (for now)