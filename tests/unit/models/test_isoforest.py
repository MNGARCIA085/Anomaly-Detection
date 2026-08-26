import numpy as np
from sklearn.ensemble import IsolationForest

from anomaly_detection.models.classic.isoforest.model import IsoWrapper


def make_data():
    """Create a small deterministic dataset for Isolation Forest tests."""
    rng = np.random.RandomState(42)
    return rng.normal(size=(20, 3))


def make_wrapper():
    """Create and fit a small Isolation Forest wrapper."""
    model = IsolationForest(
        n_estimators=10,
        random_state=42,
    )

    return IsoWrapper(model)


def test_fit_returns_wrapper():
    """fit should train the underlying model and return the wrapper."""
    X = make_data()
    wrapper = make_wrapper()

    result = wrapper.fit(X)

    assert result is wrapper
    assert hasattr(wrapper.model, "n_features_in_")


def test_get_scores_returns_one_score_per_sample():
    """Each input sample should produce exactly one anomaly score."""
    X = make_data()
    wrapper = make_wrapper()
    wrapper.fit(X)

    scores = wrapper.get_scores(X)

    assert scores.shape == (len(X),)
    assert np.isfinite(scores).all()


def test_predict_returns_binary_anomaly_labels():
    """Isolation Forest predictions should be converted to 0=normal and 1=anomaly."""
    X = make_data()
    wrapper = make_wrapper()
    wrapper.fit(X)

    predictions = wrapper.predict(X)

    assert predictions.shape == (len(X),)
    assert set(predictions).issubset({0, 1})


def test_input_dim_matches_fitted_model():
    """input_dim should expose the number of features learned by the model."""
    X = make_data()
    wrapper = make_wrapper()
    wrapper.fit(X)

    assert wrapper.input_dim == X.shape[1]


def test_save_and_load_preserves_model_behavior(tmp_path):
    """A saved and loaded wrapper should produce the same anomaly scores."""
    X = make_data()
    wrapper = make_wrapper()
    wrapper.fit(X)

    expected_scores = wrapper.get_scores(X)

    path = tmp_path / "isoforest.joblib"
    wrapper.save(path)

    loaded = IsoWrapper.load(path)
    loaded_scores = loaded.get_scores(X)

    np.testing.assert_allclose(
        loaded_scores,
        expected_scores,
    )