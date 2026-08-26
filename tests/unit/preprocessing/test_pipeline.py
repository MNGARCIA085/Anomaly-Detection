import joblib
import numpy as np

from anomaly_detection.preprocessing.pipeline import PreprocessingPipeline


class AddOne:
    """Simple transformer used to verify pipeline composition."""

    def fit(self, X):
        return self

    def transform(self, X):
        return X + 1


class MultiplyByTwo:
    """Simple transformer used to verify transformation order."""

    def fit(self, X):
        return self

    def transform(self, X):
        return X * 2


class InputTrackingTransformer:
    """Transformer used to verify the input received during fitting."""

    def __init__(self, offset=0):
        self.offset = offset
        self.fit_input = None

    def fit(self, X):
        self.fit_input = X.copy()
        return self

    def transform(self, X):
        return X + self.offset


def test_transform_applies_steps_in_order():
    """Each step should receive the output of the previous step."""
    X = np.array([[1.0], [2.0]])

    pipeline = PreprocessingPipeline([
        AddOne(),
        MultiplyByTwo(),
    ])

    result = pipeline.transform(X)

    expected = np.array([
        [4.0],
        [6.0],
    ])

    np.testing.assert_array_equal(result, expected)


def test_fit_fits_each_step_on_previous_step_output():
    """Each step should be fitted on the output of the previous step."""
    X = np.array([[1.0], [2.0]])

    first = InputTrackingTransformer(offset=10)
    second = InputTrackingTransformer(offset=20)

    pipeline = PreprocessingPipeline([first, second])

    pipeline.fit(X)

    np.testing.assert_array_equal(
        first.fit_input,
        X,
    )

    np.testing.assert_array_equal(
        second.fit_input,
        X + 10,
    )


def test_fit_transform_returns_transformed_data():
    """fit_transform should fit the pipeline and return its transformed output."""
    X = np.array([[1.0], [2.0]])

    pipeline = PreprocessingPipeline([
        AddOne(),
        MultiplyByTwo(),
    ])

    result = pipeline.fit_transform(X)

    expected = np.array([
        [4.0],
        [6.0],
    ])

    np.testing.assert_array_equal(result, expected)


def test_pipeline_can_be_saved_and_loaded(tmp_path):
    """A saved pipeline should preserve its preprocessing behavior."""
    X = np.array([[1.0], [2.0]])

    pipeline = PreprocessingPipeline([
        AddOne(),
        MultiplyByTwo(),
    ])

    pipeline.fit(X)

    path = tmp_path / "pipeline.joblib"
    pipeline.save(path)

    loaded = joblib.load(path)

    np.testing.assert_array_equal(
        loaded.transform(X),
        pipeline.transform(X),
    )