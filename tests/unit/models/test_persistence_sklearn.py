import numpy as np
from sklearn.ensemble import IsolationForest

from anomaly_detection.models.persistence.sklearn import (
    save_sklearn_model,
    load_sklearn_model,
)


def test_sklearn_model_save_and_load(tmp_path):
    """A saved sklearn model should produce the same predictions after loading."""
    X = np.array([
        [0.0, 0.0],
        [0.1, 0.2],
        [1.0, 1.0],
        [1.1, 0.9],
    ])

    model = IsolationForest(
        n_estimators=10,
        random_state=42,
    )

    model.fit(X)

    path = tmp_path / "model.joblib"

    save_sklearn_model(
        model,
        path,
    )

    loaded_model = load_sklearn_model(path)

    original_predictions = model.predict(X)
    loaded_predictions = loaded_model.predict(X)

    np.testing.assert_array_equal(
        original_predictions,
        loaded_predictions,
    )