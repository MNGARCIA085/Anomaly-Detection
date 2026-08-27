import numpy as np
import torch
import torch.nn as nn

from anomaly_detection.training.schemas import TrainingConfig
from anomaly_detection.training.trainer import NNTrainer


def test_trainer_fit_updates_model_and_records_history():
    """NNTrainer should train the model and record training and validation loss."""
    torch.manual_seed(42)

    X_train = np.random.RandomState(42).normal(
        size=(8, 3),
    ).astype(np.float32)

    X_val = np.random.RandomState(43).normal(
        size=(4, 3),
    ).astype(np.float32)

    model = nn.Linear(3, 3)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
    )

    criterion = nn.MSELoss()

    cfg = TrainingConfig(
        batch_size=4,
        epochs=2,
        device="cpu",
        shuffle=True,
        num_workers=0,
        optimizer=optimizer,
        loss=criterion,
        callbacks=[],
    )

    trainer = NNTrainer(cfg)

    initial_parameters = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    result = trainer.fit(
        model,
        X_train,
        X_val,
    )

    assert result is model
    assert trainer.history is not None

    train_losses = trainer.history.get("train_loss")
    val_losses = trainer.history.get("val_loss")

    assert len(train_losses) == 2
    assert len(val_losses) == 2

    assert all(
        np.isfinite(loss)
        for loss in train_losses
    )

    assert all(
        np.isfinite(loss)
        for loss in val_losses
    )

    assert any(
        not torch.equal(initial, current)
        for initial, current in zip(
            initial_parameters,
            model.parameters(),
        )
    )



"""
One small but meaningful unit test: use a tiny real PyTorch model, real optimizer/loss, 
and only one epoch. 
That verifies the actual training contract without turning it into an integration test.
It verifies the important observable behavior:

fit() returns the same model.
Training actually updates parameters.
Training loss is recorded.
Validation loss is recorded.
The configured number of epochs is respected.
Losses are finite.

It uses real PyTorch components, but that's still reasonable for a unit test 
here because we're testing NNTrainer's behavior, 
not mocking away the very operations it is responsible for.

"""