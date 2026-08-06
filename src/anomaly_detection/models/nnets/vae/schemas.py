from dataclasses import dataclass
from typing import List


@dataclass
class VAEConfig:
    input_dim: int

    encoder_dims: List[int] = (8, 4)
    latent_dim: int = 2
    decoder_dims: List[int] = (4, 8)

    beta: float = 1.0