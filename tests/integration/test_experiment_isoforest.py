import numpy as np

from anomaly_detection.evaluation.evaluator import Evaluator
from anomaly_detection.experiments.experiments import Experiment
from anomaly_detection.infra.logging.null_logger import NullLogger


def test_experiment_runs_isolation_forest_end_to_end():
    """Experiment should run the complete Isolation Forest workflow."""
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
            "feature_selection": {
                "enabled": False,
            },
            "scaler": {
                "name": "standard",
                "params": {
                    "with_mean": True,
                    "with_std": True,
                },
            },
            "dimensionality": {
                "enabled": False,
            },
        },
        "data": {
            "windowing": {
                "size": 5,
            },
        },
        "models": {
            "n_estimators": 20,
            "contamination": 0.1,
        },
    }

    experiment = Experiment(
        model_type="isoforest",
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
IsoForest
  → native prediction
  → no thresholding
  → flattened input
"""