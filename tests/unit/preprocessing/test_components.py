import pytest

from anomaly_detection.preprocessing.components.scalers import (
    create_scaler,
)
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
)


def test_create_scaler_returns_requested_scaler():
    scaler = create_scaler("standard")

    assert isinstance(scaler, StandardScaler)


def test_create_scaler_passes_parameters():
    scaler = create_scaler(
        "standard",
        with_mean=False,
        with_std=True,
    )

    assert scaler.with_mean is False
    assert scaler.with_std is True


def test_create_scaler_rejects_unknown_scaler():
    with pytest.raises(
        ValueError,
        match="Unknown scaler: unknown",
    ):
        create_scaler("unknown")



"""
Correct registry resolution → verifies "standard" maps to the correct class.
Parameter forwarding → verifies your factory correctly passes **params.
Invalid name → verifies the public error contract.
"""