from dataclasses import dataclass

import torch
import torch.nn as nn

from anomaly_detection.models.persistence.torch import (
    save_torch_model,
    load_torch_model,
)


@dataclass
class SimpleModelConfig:
    input_dim: int = 3
    output_dim: int = 2


class SimpleModel(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.linear = nn.Linear(
            config.input_dim,
            config.output_dim,
        )

    def forward(self, X):
        return self.linear(X)


def test_torch_model_save_and_load(tmp_path):
    """A saved PyTorch model should preserve its configuration and weights."""
    torch.manual_seed(42)

    config = SimpleModelConfig()

    model = SimpleModel(config)
    model.eval()

    X = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )

    path = tmp_path / "model"

    save_torch_model(
        model,
        path,
    )

    loaded_model = load_torch_model(
        SimpleModel,
        path,
    )

    assert loaded_model.config == config

    with torch.no_grad():
        original_output = model(X)
        loaded_output = loaded_model(X)

    torch.testing.assert_close(
        original_output,
        loaded_output,
    )