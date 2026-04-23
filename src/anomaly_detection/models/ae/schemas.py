from dataclasses import dataclass
from typing import List


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
