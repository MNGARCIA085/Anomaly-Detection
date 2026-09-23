import numpy as np

from anomaly_detection.evaluation.evaluator import Evaluator
from anomaly_detection.experiments.experiments import Experiment
from anomaly_detection.infra.logging.null_logger import NullLogger


class TrackingThreshold:

    def __init__(self):
        self.fit_scores = None

    def fit(self, train_scores, val_scores=None, y_val=None):
        self.fit_scores = np.asarray(train_scores).copy()
        return self

    def get_threshold(self):
        return 0.5


class TrackingThresholding:

    def __init__(self, config):
        self.strategy = TrackingThreshold()

    def fit(self, train_scores, val_scores=None, y_val=None):
        self.strategy.fit(train_scores, val_scores, y_val)
        return self

    def get_threshold(self):
        return self.strategy.get_threshold()


class DummyWrapper:

    def fit(self, X_train, X_val):
        return self

    def get_scores(self, X):
        # Different scores for train and validation.
        return X[:, 0]

    def predict(self, X, threshold):
        scores = self.get_scores(X)
        return (scores > threshold).astype(int)

    @property
    def history(self):
        return None


class DummyEntry:

    def build_preprocessor(self, cfg):
        return IdentityPreprocessor()

    def adapt_input(self, X):
        return X.reshape(X.shape[0], -1)

    def build(self, model_cfg, training_cfg=None, input_shape=None):
        return DummyWrapper()


class IdentityPreprocessor:

    def fit_transform(self, X):
        return X

    def transform(self, X):
        return X


def test_experiment_fits_threshold_on_training_scores(monkeypatch):
    """Experiment should fit the threshold using training scores, not validation scores."""
    import anomaly_detection.models.registry as registry

    monkeypatch.setitem(
        registry.MODEL_REGISTRY,
        "dummy",
        lambda: DummyEntry(),
    )

    thresholding = TrackingThresholding(
        {
            "name": "tracking",
        }
    )

    monkeypatch.setattr(
        "anomaly_detection.experiments.experiments.Thresholding",
        lambda config: thresholding,
    )

    X_train = np.array([
        [0.1],
        [0.2],
        [0.3],
        [0.4],
    ])

    X_val = np.array([
        [10.0],
        [20.0],
        [30.0],
        [40.0],
    ])

    y_val = np.zeros(len(X_val), dtype=int)

    cfg = {
        "prep": {},
        "data": {
            "windowing": {
                "size": 1,
            },
        },
        "thresholding": {
            "name": "tracking",
        },
    }

    experiment = Experiment(
        model_type="dummy",
        evaluator=Evaluator(),
        logger=NullLogger(),
    )

    experiment.run(
        cfg=cfg,
        X_train=X_train,
        X_val=X_val,
        y_val=y_val,
    )

    np.testing.assert_array_equal(
        thresholding.strategy.fit_scores,
        X_train[:, 0],
    )