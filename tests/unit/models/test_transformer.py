


"""
TESTS:
	positional encoding changes/adds the expected representation;
	TransformerAE preserves input shape;
	wrapper produces one anomaly score per window;
	wrapper predictions respect the threshold;
	entry builds the correct model/input contract.
"""


import numpy as np
import torch

from anomaly_detection.models.nnets.transformer.model import (
    PositionalEncoding,
    TransformerAE,
    TransformerAEWrapper,
)
from anomaly_detection.models.nnets.transformer.schemas import TransformerAEConfig


def make_config():
    """Create a small Transformer configuration suitable for fast tests."""
    return TransformerAEConfig(
        input_dim=3,
        seq_len=4,
        d_model=8,
        nhead=2,
        num_encoder_layers=1,
        dim_feedforward=16,
        dropout=0.0,
    )


def make_model():
    """Create a small Transformer autoencoder for testing."""
    return TransformerAE(make_config())


def test_positional_encoding_preserves_shape():
    """Positional encoding should preserve the input tensor shape."""
    encoding = PositionalEncoding(
        d_model=8,
        seq_len=4,
    )

    X = torch.zeros(2, 4, 8)

    result = encoding(X)

    assert result.shape == X.shape


def test_transformer_autoencoder_preserves_input_shape():
    """TransformerAE should reconstruct data with the same shape as its input."""
    model = make_model()

    X = torch.randn(2, 4, 3)

    result = model(X)

    assert result.shape == X.shape


def test_wrapper_get_scores_returns_one_score_per_window():
    """Each input window should produce exactly one anomaly score."""
    model = make_model()
    wrapper = TransformerAEWrapper(
        model=model,
        trainer=None,
    )

    X = np.random.randn(5, 4, 3).astype(np.float32)

    scores = wrapper.get_scores(X)

    assert scores.shape == (5,)
    assert np.isfinite(scores).all()


def test_wrapper_predicts_anomalies_above_threshold():
    """Predictions should classify scores strictly greater than the threshold."""
    model = make_model()
    wrapper = TransformerAEWrapper(
        model=model,
        trainer=None,
    )

    X = np.random.randn(5, 4, 3).astype(np.float32)

    scores = wrapper.get_scores(X)
    threshold = float(np.median(scores))

    predictions = wrapper.predict(
        X,
        threshold=threshold,
    )

    expected = (scores > threshold).astype(int)

    np.testing.assert_array_equal(
        predictions,
        expected,
    )


def test_wrapper_input_dim_matches_model_configuration():
    """The wrapper should expose the model's configured feature dimension."""
    model = make_model()
    wrapper = TransformerAEWrapper(
        model=model,
        trainer=None,
    )

    assert wrapper.input_dim == 3




"""
The contract were protecting here is:
(N, T, F)
   ↓
TransformerAE
   ↓
(N, T, F)
   ↓
reconstruction error
   ↓
(N,)
   ↓
threshold
   ↓
(N,)
"""



"""
NOTE

	I wouldn't test TransformerEntry yet in this file. 
	Its build() method is orchestration across:

	config
	 ↓
	optimizer
	 ↓
	loss
	 ↓
	callbacks
	 ↓
	trainer
	 ↓
	TransformerAEWrapper
"""