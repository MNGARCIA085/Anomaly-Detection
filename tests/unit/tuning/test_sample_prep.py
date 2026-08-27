from types import SimpleNamespace

from anomaly_detection.tuning.sample_prep import (
    sample_window_size,
    sample_scaler,
)


class FakeTrial:
    def suggest_categorical(self, name, choices):
        return choices[0]


def test_sample_window_size_returns_selected_value():
    """sample_window_size should return a value from the configured choices."""
    trial = FakeTrial()

    cfg = SimpleNamespace(
        choices=[1, 5, 10, 20],
    )

    result = sample_window_size(trial, cfg)

    assert result == 1


def test_sample_scaler_returns_selected_scaler_config():
    """sample_scaler should return the selected scaler name and empty parameters."""
    trial = FakeTrial()

    cfg = SimpleNamespace(
        names=["standard", "minmax"],
    )

    result = sample_scaler(trial, cfg)

    assert result == {
        "name": "standard",
        "params": {},
    }