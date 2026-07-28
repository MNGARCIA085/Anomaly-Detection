from dataclasses import dataclass, field
from typing import List,Any

@dataclass
class TrainState:
    epoch: int = 0
    train_loss: float = 0.0
    val_loss: float = None
    model: Any = None
    stop_training: bool = False




@dataclass
class TrainingConfig:
    lr: float
    batch_size: int
    epochs: int
    
    device: str = "cpu"
    shuffle: bool = True
    num_workers: int = 0

    callbacks: List[Any] = field(default_factory=list)


"""
@dataclass
class TrainingConfig:
    lr: float
    batch_size: int
    epochs: int

    optimizer: str = "adam"
    weight_decay: float = 0.0

    device: str = "cpu"
    shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = False

    callbacks: list[Any] = field(default_factory=list)
"""