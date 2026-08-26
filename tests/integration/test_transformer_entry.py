import numpy as np

from anomaly_detection.models.nnets.transformer.entry import TransformerEntry
from anomaly_detection.models.nnets.transformer.model import TransformerAEWrapper


def test_transformer_entry_builds_model_from_configuration():
    """TransformerEntry should integrate configuration with the model and training components."""
    entry = TransformerEntry()

    model_cfg = {
        "d_model": 8,
        "nhead": 2,
        "num_encoder_layers": 1,
        "dim_feedforward": 16,
        "dropout": 0.0,
    }

    training_cfg = {
        "batch_size": 2,
        "epochs": 1,
        "type": "default",
        "optimizer": {
            "name": "adam",
            "params": {
                "lr": 0.001,
                "betas": [0.9, 0.999],
            },
        },
        "loss": {
            "name": "mse",
        },
        "callbacks": [
            {
                "name": "print_loss",
            },
            {
                "name": "early_stopping",
                "params": {
                    "patience": 5,
                },
            },
        ],
    }

    X = np.random.randn(4, 5, 3).astype(np.float32)

    model = entry.build(
        cfg_model=model_cfg,
        cfg_training=training_cfg,
        input_shape=X.shape,
    )

    assert isinstance(model, TransformerAEWrapper)
    assert model.input_dim == 3



"""

We're not mocking those dependencies, 
so we're testing that they work together through the Entry interface


And importantly, it doesn't train anything. 
That's intentional—we'll test the training in training/

It is integration because TransformerEntry.build() 
invokes and connects multiple real components:

TransformerEntry
    ↓
TransformerAEConfig
    ↓
TransformerAE
    ↓
optimizer factory
loss factory
callback factory
trainer registry
    ↓
TransformerAEWrapper


(is more like component integration test)

"""