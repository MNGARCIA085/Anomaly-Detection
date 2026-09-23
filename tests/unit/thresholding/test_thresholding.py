import numpy as np
import pytest

from anomaly_detection.thresholding.strategies import QuantileThreshold
from anomaly_detection.thresholding.registry import (
    create_threshold_strategy,
)
from anomaly_detection.thresholding.thresholding import Thresholding


def test_quantile_threshold_computes_threshold():
    """QuantileThreshold should compute the configured quantile from scores."""
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    strategy = QuantileThreshold(quantile=0.8)

    strategy.fit(scores)

    assert strategy.get_threshold() == pytest.approx(0.42)


def test_quantile_threshold_requires_fit():
    """Getting a threshold before fitting should raise an error."""
    strategy = QuantileThreshold()

    with pytest.raises(
        RuntimeError,
        match="Threshold has not been fitted",
    ):
        strategy.get_threshold()


def test_create_threshold_strategy_returns_registered_strategy():
    """Factory should create the strategy registered under the requested name."""
    strategy = create_threshold_strategy(
        "quantile",
        quantile=0.95,
    )

    assert isinstance(strategy, QuantileThreshold)
    assert strategy.quantile == 0.95


def test_thresholding_without_config_returns_no_threshold():
    """Thresholding without configuration should behave as an optional threshold."""
    thresholding = Thresholding(config=None)

    train_scores = np.array([0.1, 0.2, 0.3])
    val_scores = np.array([0.4, 0.5])
    y_val = np.array([0, 1])

    thresholding.fit(train_scores, val_scores, y_val)

    assert thresholding.get_threshold() is None


def test_thresholding_can_save_and_load(tmp_path):
    """A fitted Thresholding object should preserve its threshold after persistence."""
    train_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    val_scores = np.array([0.6, 0.7])
    y_val = np.array([0, 1])

    thresholding = Thresholding(
        config={
            "name": "quantile",
            "params": {"quantile": 0.8},
        }
    )

    thresholding.fit(train_scores, val_scores, y_val)
    expected_threshold = thresholding.get_threshold()

    path = tmp_path / "thresholding.joblib"
    thresholding.save(path)

    loaded = Thresholding.load(path)

    assert loaded.get_threshold() == pytest.approx(expected_threshold)