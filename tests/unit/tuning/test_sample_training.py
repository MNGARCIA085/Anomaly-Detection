from types import SimpleNamespace

from omegaconf import OmegaConf

from anomaly_detection.tuning.sample_training import (
    sample_callbacks,
    sample_optimizer,
)


class FakeTrial:

    def __init__(self, categorical_values=None):
        self.categorical_values = categorical_values or {}

    def suggest_categorical(self, name, choices):
        return self.categorical_values.get(
            name,
            choices[0],
        )

    def suggest_int(self, name, low, high):
        return low

    def suggest_float(self, name, low, high, log=False):
        return low


def test_sample_callbacks_returns_enabled_callbacks():
    """sample_callbacks should create callbacks enabled by the tuning configuration."""
    trial = FakeTrial(
        {
            "training.callbacks.print_loss": True,
            "training.callbacks.early_stopping.enabled": True,
        }
    )

    cfg = SimpleNamespace(
        print_loss=True,
        early_stopping=SimpleNamespace(
            enabled=True,
            patience=SimpleNamespace(
                low=2,
                high=5,
            ),
        ),
    )

    result = sample_callbacks(trial, cfg)

    assert result == [
        {
            "name": "print_loss",
            "params": {},
        },
        {
            "name": "early_stopping",
            "params": {
                "patience": 2,
            },
        },
    ]


def test_sample_callbacks_returns_empty_when_disabled():
    """sample_callbacks should return no callbacks when callback tuning is disabled."""
    trial = FakeTrial()

    cfg = SimpleNamespace(
        print_loss=False,
        early_stopping=SimpleNamespace(
            enabled=False,
            patience=SimpleNamespace(
                low=2,
                high=5,
            ),
        ),
    )

    result = sample_callbacks(trial, cfg)

    assert result == []


def test_sample_optimizer_delegates_to_selected_optimizer():
    """sample_optimizer should delegate sampling to the selected optimizer."""
    trial = FakeTrial(
        {
            "optimizer.name": "adam",
        }
    )

    cfg = OmegaConf.create({
        "names": ["adam"],
        "adam": {
            "lr": {
                "low": 0.001,
                "high": 0.01,
                "log": True,
            },
            "betas": {
                "choices": [
                    [0.9, 0.999],
                ],
            },
        },
    })

    result = sample_optimizer(trial, cfg)

    assert result["name"] == "adam"
    assert result["params"]["lr"] == 0.001
    assert result["params"]["betas"] == [0.9, 0.999]