from anomaly_detection.training.callbacks.implementations import (
    EarlyStopping,
)
from anomaly_detection.training.callbacks.registry import (
    create_callbacks,
)
from anomaly_detection.training.schemas import TrainState


def test_early_stopping_resets_counter_when_loss_improves():
    """EarlyStopping should reset its counter whenever validation loss improves."""
    callback = EarlyStopping(patience=2)

    state = TrainState(model=None)

    state.val_loss = 1.0
    callback.on_epoch_end(state)

    assert callback.best == 1.0
    assert callback.counter == 0
    assert state.stop_training is False

    state.val_loss = 0.8
    callback.on_epoch_end(state)

    assert callback.best == 0.8
    assert callback.counter == 0
    assert state.stop_training is False


def test_early_stopping_stops_after_patience():
    """EarlyStopping should request training termination after patience is exceeded."""
    callback = EarlyStopping(patience=2)

    state = TrainState(model=None)

    state.val_loss = 1.0
    callback.on_epoch_end(state)

    state.val_loss = 1.1
    callback.on_epoch_end(state)

    assert callback.counter == 1
    assert state.stop_training is False

    state.val_loss = 1.2
    callback.on_epoch_end(state)

    assert callback.counter == 2
    assert state.stop_training is True


def test_create_callbacks_from_configuration():
    """create_callbacks should instantiate callbacks from configuration."""
    cfg = [
        {
            "name": "early_stopping",
            "params": {
                "patience": 3,
            },
        },
    ]

    callbacks = create_callbacks(cfg)

    assert len(callbacks) == 1
    assert isinstance(callbacks[0], EarlyStopping)
    assert callbacks[0].patience == 3