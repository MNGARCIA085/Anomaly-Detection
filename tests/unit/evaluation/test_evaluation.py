import numpy as np

from anomaly_detection.evaluation.evaluator import Evaluator


def test_evaluate_scores_only_returns_mean_score():
    """Evaluation without labels should return only the mean score."""
    scores = np.array([0.1, 0.2, 0.3, 0.4])

    result = Evaluator().evaluate(scores)

    assert result == {
        "mean_score": 0.25,
    }


def test_evaluate_with_labels_returns_threshold_independent_metrics():
    """Evaluation with labels should include ROC-AUC and PR-AUC."""
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    y_true = np.array([0, 0, 1, 1])

    result = Evaluator().evaluate(
        scores=scores,
        y_true=y_true,
    )

    assert result["mean_score"] == 0.5
    assert result["auc"] == 1.0
    assert result["pr_auc"] == 1.0


def test_evaluate_with_predictions_returns_classification_metrics():
    """Evaluation with predictions should include classification metrics and confusion matrix."""
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    y_true = np.array([0, 0, 1, 1])
    predictions = np.array([0, 1, 1, 1])

    result = Evaluator().evaluate(
        scores=scores,
        y_true=y_true,
        predictions=predictions,
    )

    assert result["precision"] == 2 / 3
    assert result["recall"] == 1.0
    assert result["f1"] == 0.8

    assert result["tn"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 0
    assert result["tp"] == 2


def test_evaluate_handles_no_predicted_positives():
    """Precision should safely return zero when there are no predicted positives."""
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    y_true = np.array([0, 0, 1, 1])
    predictions = np.array([0, 0, 0, 0])

    result = Evaluator().evaluate(
        scores=scores,
        y_true=y_true,
        predictions=predictions,
    )

    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0

    assert result["tn"] == 2
    assert result["fp"] == 0
    assert result["fn"] == 2
    assert result["tp"] == 0


"""
scores
  │
  └── mean_score

scores + y_true
  │
  ├── mean_score
  ├── AUC
  └── PR-AUC

scores + y_true + predictions
  │
  ├── threshold-independent metrics
  └── threshold-dependent metrics
      ├── precision
      ├── recall
      ├── F1
      └── confusion matrix
"""