from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class IntParam:
    name: str
    low: int
    high: int


@dataclass
class FloatParam:
    name: str
    low: float
    high: float
    log: bool = False


@dataclass
class CategoricalParam:
    name: str
    choices: List[Any]