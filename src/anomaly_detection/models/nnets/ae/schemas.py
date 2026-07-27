from dataclasses import dataclass, field
from typing import List, Any
from anomaly_detection.models.schemas import IntParam, FloatParam, CategoricalParam




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
    device: str = "cpu"
    callbacks: List[Any] = field(default_factory=list)
    shuffle: bool = True
    num_workers: int = 0





