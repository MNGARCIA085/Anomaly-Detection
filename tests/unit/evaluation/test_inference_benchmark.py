import numpy as np

from anomaly_detection.inference.benchmarking import (
    InferenceBenchmark,
)


class DummyModel:
    """Minimal model used to verify prediction calls."""

    def __init__(self):
        self.predict_calls = 0

    def predict(self, X):
        self.predict_calls += 1
        return np.zeros(len(X))


def test_measure_warms_up_and_repeats_prediction():
    """Benchmark should perform requested warm-up predictions plus the requested repetitions."""
    model = DummyModel()
    X = np.zeros((5, 2))
    warmup = 3
    repetitions = 3

    result = InferenceBenchmark().measure(
        runner=model,
        X=X,
        repetitions=repetitions,
        warmup=warmup,
    )

    # Total calls must equal warm-up calls + repetitions
    assert model.predict_calls == warmup + repetitions
    assert result["total_seconds"] >= 0
    assert result["avg_ms"] >= 0


def test_measure_uses_default_warmup_and_repetitions():
    """Benchmark should use default parameter values if none are provided."""
    model = DummyModel()
    X = np.zeros((5, 2))

    InferenceBenchmark().measure(
        runner=model,
        X=X,
    )

    # Default warmup=5 + default repetitions=20
    assert model.predict_calls == 25


def test_measure_returns_consistent_timing_metrics():
    """Average latency should equal total elapsed time divided by repetitions."""
    model = DummyModel()
    X = np.zeros((5, 2))
    repetitions = 5

    result = InferenceBenchmark().measure(
        runner=model,
        X=X,
        repetitions=repetitions,
    )

    expected_avg_ms = (
        result["total_seconds"] / repetitions * 1000
    )

    assert result["avg_ms"] == expected_avg_ms