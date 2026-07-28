from dataclasses import dataclass, field
from typing import List, Any
from anomaly_detection.models.schemas import IntParam, FloatParam, CategoricalParam




@dataclass
class AEConfig:
    input_dim: int
    encoder_dims: List[int] = (8, 4)
    decoder_dims: List[int] = (8,)

