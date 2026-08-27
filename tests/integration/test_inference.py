import numpy as np

from anomaly_detection.inference.runner import InferenceRunner


class TrackingPreprocessor:

    def __init__(self):
        self.called = False

    def transform(self, X):
        self.called = True
        return X + 1


class TrackingWindowing:

    def __init__(self):
        self.called = False

    def transform(self, X):
        self.called = True
        return X[:, :2]


class TrackingTemporalPreprocessor:

    def __init__(self):
        self.called = False

    def transform(self, X):
        self.called = True
        return X * 2


class TrackingEntry:

    def __init__(self):
        self.called = False

    def adapt_input(self, X):
        self.called = True
        return X.reshape(X.shape[0], -1)


class TrackingThresholding:

    def __init__(self):
        self.called = False

    def get_threshold(self):
        self.called = True
        return 0.5


class TrackingWrapper:

    def __init__(self):
        self.called_with = None

    def predict(self, X, threshold):
        self.called_with = (X, threshold)
        return np.ones(X.shape[0], dtype=int)


def test_inference_runner_executes_full_pipeline():
    """InferenceRunner should apply preprocessing, windowing, temporal preprocessing, adaptation, and thresholding in order."""
    prep = TrackingPreprocessor()
    windowing = TrackingWindowing()
    temporal_prep = TrackingTemporalPreprocessor()
    entry = TrackingEntry()
    thresholding = TrackingThresholding()
    wrapper = TrackingWrapper()

    runner = InferenceRunner(
        prep=prep,
        windowing=windowing,
        entry=entry,
        wrapper=wrapper,
        temporal_prep=temporal_prep,
        thresholding=thresholding,
    )

    X = np.ones((4, 3))

    predictions = runner.predict(X)

    assert prep.called
    assert windowing.called
    assert temporal_prep.called
    assert entry.called
    assert thresholding.called

    assert wrapper.called_with is not None

    _, threshold = wrapper.called_with

    assert threshold == 0.5
    assert len(predictions) == 4



"""
The main thing this protects is your inference contract:

    X
     ↓
    preprocessor.transform()
     ↓
    windowing.transform()
     ↓
    window_level_temporal_prep.transform()  (optional)
     ↓
    entry.adapt_input()
     ↓
    thresholding.get_threshold()    (optional)
     ↓
    wrapper.predict()

    This is particularly valuable because ordering matters her
"""