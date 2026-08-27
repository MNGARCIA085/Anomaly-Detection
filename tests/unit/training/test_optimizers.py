import torch
import pytest

from anomaly_detection.training.optimizers.implementations import (
    AdamOptimizer,
)
from anomaly_detection.training.optimizers.registry import (
    create_optimizer,
)


def test_create_optimizer_builds_adam_with_config():
    """create_optimizer should instantiate Adam using the supplied parameters."""
    model = torch.nn.Linear(3, 2)

    cfg = {
        "name": "adam",
        "params": {
            "lr": 0.001,
            "betas": (0.9, 0.999),
        },
    }

    optimizer = create_optimizer(
        cfg,
        model.parameters(),
    )

    assert isinstance(
        optimizer,
        torch.optim.Adam,
    )

    assert optimizer.defaults["lr"] == 0.001
    assert optimizer.defaults["betas"] == (0.9, 0.999)


def test_create_optimizer_rejects_unknown_optimizer():
    """create_optimizer should raise an error for an unknown optimizer."""
    model = torch.nn.Linear(3, 2)

    cfg = {
        "name": "unknown",
        "params": {},
    }

    with pytest.raises(KeyError):
        create_optimizer(
            cfg,
            model.parameters(),
        )