import numpy as np

from anomaly_detection.evaluation.inference_benchmark import (
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
    """Benchmark should perform one warm-up prediction plus the requested repetitions."""
    model = DummyModel()
    X = np.zeros((5, 2))

    result = InferenceBenchmark().measure(
        model=model,
        X=X,
        repetitions=3,
    )

    assert model.predict_calls == 4
    assert result["total_seconds"] >= 0
    assert result["avg_ms"] >= 0


def test_measure_returns_consistent_timing_metrics():
    """Average latency should equal total elapsed time divided by repetitions."""
    model = DummyModel()
    X = np.zeros((5, 2))
    repetitions = 5

    result = InferenceBenchmark().measure(
        model=model,
        X=X,
        repetitions=repetitions,
    )

    expected_avg_ms = (
        result["total_seconds"] / repetitions * 1000
    )

    assert result["avg_ms"] == expected_avg_ms



"""
The timing itself is inherently nondeterministic, 
so we should test the contract, not exact performance.

The first test protects the warm-up + repetition contract. 
The second protects the calculation of avg_ms.

I would not test things like avg_ms < 10, because that would make 
the test dependent on the machine running it.
"""