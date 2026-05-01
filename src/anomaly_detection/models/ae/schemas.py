from dataclasses import dataclass, field
from typing import List, Any
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
    # new
    device: str = "cpu"
    callbacks: List[Any] = field(default_factory=list)
    # for dataclass
    shuffle: bool = True
    num_workers: int = 0


@dataclass
class TrainState:
    epoch: int = 0
    loss: float = 0.0
    model: Any = None
    stop_training: bool = False





# Tuning
@dataclass
class AEModelTuningConfig:
    n_layers: IntParam
    encoder_dim: IntParam


@dataclass
class AETrainingTuningConfig:
    lr: FloatParam
    batch_size: CategoricalParam
    epochs: int  # fixed, not tuned (for now)
    #
    callbacks: List[Any] = field(default_factory=list)
    # for dataclass
    shuffle: bool = True
    num_workers: int = 0


@dataclass
class AETuningConfig:
    model_space: AEModelTuningConfig
    training_space: AETrainingTuningConfig