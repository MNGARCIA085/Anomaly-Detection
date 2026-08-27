import numpy as np

from anomaly_detection.evaluation.evaluator import Evaluator
from anomaly_detection.experiments.experiments import Experiment
from anomaly_detection.infra.logging.null_logger import NullLogger


import numpy as np

from anomaly_detection.evaluation.evaluator import Evaluator
from anomaly_detection.experiments.experiments import Experiment
from anomaly_detection.infra.logging.null_logger import NullLogger


# Dummy preprocessor to track data passed to fit_transform and transform calls
class TrackingPreprocessor:

    def __init__(self):
        # Stores the dataset passed during the fit step (should be X_train)
        self.fit_data = None
        # Stores a list of datasets passed during transform steps (should capture X_val)
        self.transform_data = []

    def fit_transform(self, X):
        # Save a copy of the training data to prevent in-place mutation side-effects
        self.fit_data = X.copy()
        return X

    def transform(self, X):
        # Append transformed subsets (e.g., validation data) to track execution history
        self.transform_data.append(X.copy())
        return X


# Registry entry mock providing custom preprocessor and model creation logic
class TrackingEntry:

    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

    def build_preprocessor(self, cfg):
        # Returns the spy preprocessor instance being monitored by the test
        return self.preprocessor

    def adapt_input(self, X):
        # Flattens input dimensions if needed by downstream models
        return X.reshape(X.shape[0], -1)

    def build(self, model_cfg, training_cfg=None, input_shape=None):
        # Instantiate dummy model wrapper for isolation
        return DummyWrapper()


# Minimal stub model to satisfy the Experiment runner pipeline
class DummyWrapper:

    def fit(self, X_train, X_val):
        # No-op fit step; returns self to allow method chaining
        return self

    def get_scores(self, X):
        # Returns dummy zero anomaly scores for input instances
        return np.zeros(len(X))

    def predict(self, X):
        # Returns dummy zero binary predictions (normal class)
        return np.zeros(len(X), dtype=int)

    @property
    def history(self):
        # Satisfies optional training history interface expected by Evaluator/Logger
        return None


def test_experiment_fits_preprocessor_only_on_train(monkeypatch):
    """Experiment should fit preprocessing on training data and only transform validation data."""
    # Instantiate spy preprocessor to verify method invocation details
    preprocessor = TrackingPreprocessor()

    # Wrap spy preprocessor in a mock model entry builder
    entry = TrackingEntry(preprocessor)

    # Inject mock entry into the global MODEL_REGISTRY via Pytest monkeypatch
    monkeypatch.setitem(
        __import__(
            "anomaly_detection.models.registry",
            fromlist=["MODEL_REGISTRY"],
        ).MODEL_REGISTRY,
        "tracking",
        lambda: entry,
    )

    # Generate synthetic training and validation feature matrices
    X_train = np.arange(30).reshape(10, 3)
    X_val = np.arange(30, 60).reshape(10, 3)

    # Generate synthetic validation ground truth labels
    y_val = np.zeros(len(X_val), dtype=int)

    # Define minimal test configuration dictionary
    cfg = {
        "prep": {},
        "data": {
            "windowing": {
                "size": 2,
            },
        },
    }



    """
    By passing a mock (TrackingPreprocessor) into experiment.run(), 
    the test acts as an inspector listening to how 
    Experiment interacts with preprocessing components during execution
    """

    # Initialize experiment under test using NullLogger to prevent side-effect logging
    experiment = Experiment(
        model_type="tracking",
        evaluator=Evaluator(),
        logger=NullLogger(),
    )

    # Execute full training and validation lifecycle pipeline
    experiment.run(
        cfg=cfg,
        X_train=X_train,
        X_val=X_val,
        y_val=y_val,
    )

    # Assert preprocessor was fitted exclusively using X_train (prevents data leakage)
    np.testing.assert_array_equal(
        preprocessor.fit_data,
        X_train,
    )

    # Assert preprocessor transform was called exactly once throughout evaluation
    assert len(preprocessor.transform_data) == 1

    # Assert preprocessor transformed X_val during evaluation/inference phase
    np.testing.assert_array_equal(
        preprocessor.transform_data[0],
        X_val,
    )



"""
There is one subtle point here: Experiment also performs windowing after preprocessing, 
so the validation data passed to the preprocessor should be exactly the original X_val. 
That's what this test verifies.

This is a good integration test because it protects an important ML invariant—no validation 
leakage—without depending on a particular scaler implementation.
"""