from anomaly_detection.models import register_models
from anomaly_detection.models.registry import MODEL_REGISTRY


def test_registered_models_are_available():
    """Core model families should be registered and available by name."""
    assert "isoforest" in MODEL_REGISTRY
    assert "transformer" in MODEL_REGISTRY