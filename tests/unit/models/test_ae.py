import numpy as np
import torch

from anomaly_detection.models.nnets.ae.model import AE, AEWrapper
from anomaly_detection.models.nnets.ae.schemas import AEConfig


def make_config():
    return AEConfig(
        input_dim=4,
        encoder_dims=[3, 2],
        decoder_dims=[3],
    )


def test_ae_output_has_same_shape_as_input():
    """The autoencoder should reconstruct inputs with the original shape."""
    model = AE(make_config())

    X = torch.randn(5, 4)

    output = model(X)

    assert output.shape == X.shape


def test_ae_wrapper_get_scores_returns_one_score_per_sample():
    """Each input sample should produce exactly one reconstruction-error score."""
    model = AE(make_config())
    model.eval()

    wrapper = AEWrapper(
        model=model,
        trainer=None,
    )

    X = np.random.randn(6, 4).astype(np.float32)

    scores = wrapper.get_scores(X)

    assert scores.shape == (6,)
    assert np.all(scores >= 0)


def test_ae_wrapper_predict_applies_threshold():
    """Predictions should be 1 only when the reconstruction error exceeds the threshold."""
    model = AE(make_config())
    model.eval()

    wrapper = AEWrapper(
        model=model,
        trainer=None,
    )

    X = np.random.randn(6, 4).astype(np.float32)

    scores = wrapper.get_scores(X)
    threshold = np.median(scores)

    predictions = wrapper.predict(
        X,
        threshold,
    )

    expected = (scores > threshold).astype(int)

    np.testing.assert_array_equal(
        predictions,
        expected,
    )


def test_ae_wrapper_input_dim():
    """The wrapper should expose the input dimension from the model configuration."""
    model = AE(make_config())

    wrapper = AEWrapper(
        model=model,
        trainer=None,
    )

    assert wrapper.input_dim == 4