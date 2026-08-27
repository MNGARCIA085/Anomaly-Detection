import numpy as np

from anomaly_detection.evaluation.evaluator import Evaluator
from anomaly_detection.experiments.experiments import Experiment
from anomaly_detection.infra.logging.null_logger import NullLogger


def test_experiment_runs_transformer_end_to_end():
    """Experiment should run the Transformer workflow with temporal preprocessing and thresholding."""
    rng = np.random.RandomState(42)

    X_train = rng.normal(
        size=(50, 3),
    )

    X_val = rng.normal(
        size=(30, 3),
    )

    y_val = np.zeros(
        len(X_val),
        dtype=int,
    )
    y_val[-5:] = 1

    cfg = {
        "prep": {
            "scaler": {
                "name": "standard",
                "params": {
                    "with_mean": True,
                    "with_std": True,
                },
            },
            "temporal": {
                "delta": True,
            },
        },
        "data": {
            "windowing": {
                "size": 5,
            },
        },
        "models": {
            "d_model": 8,
            "nhead": 2,
            "num_encoder_layers": 1,
            "dim_feedforward": 16,
            "dropout": 0.0,
        },
        "training": {
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
            "callbacks": [],
        },
        "thresholding": {
            "name": "quantile",
            "params": {
                "quantile": 0.95,
            },
        },
    }

    experiment = Experiment(
        model_type="transformer",
        evaluator=Evaluator(),
        logger=NullLogger(),
    )

    metrics = experiment.run(
        cfg=cfg,
        X_train=X_train,
        X_val=X_val,
        y_val=y_val,
    )



    assert "mean_score" in metrics
    assert "auc" in metrics
    assert "pr_auc" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics

    assert 0.0 <= metrics["auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0


"""
Transformer
  → sequence input
  → temporal preprocessing
  → threshold learned from train scores
  → threshold-based prediction
"""