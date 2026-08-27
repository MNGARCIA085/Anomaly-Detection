from unittest.mock import MagicMock, patch

from anomaly_detection.tuning.tuner import Tuner

from types import SimpleNamespace

def test_tuner_run_maximizes_auc_when_labels_are_available():
    """Tuner should maximize AUC and store each trial configuration."""

    fake_entry = MagicMock()

    fake_entry.sample.side_effect = [
        {"name": "config_0"},
        {"name": "config_1"},
        {"name": "config_2"},
    ]

    fake_exp = MagicMock()

    fake_exp.run.side_effect = [
        {"auc": 0.80},
        {"auc": 0.90},
        {"auc": 0.85},
    ]

    with patch(
        "anomaly_detection.tuning.tuner.Experiment",
        return_value=fake_exp,
    ):
        tuner = Tuner(
            model_type="isoforest",
            evaluator=MagicMock(),
            tun_cfg=MagicMock(),
            logger=MagicMock(),
        )

        with patch(
            "anomaly_detection.tuning.tuner.MODEL_REGISTRY",
            {"isoforest": lambda: fake_entry},
        ):
            study = tuner.run(
                X_train=[[1]],
                X_val=[[2]],
                y_val=[0],
                n_trials=3,
            )

    assert study.direction.name == "MAXIMIZE"
    assert study.best_value == 0.90

    assert fake_exp.run.call_count == 3

    assert tuner.trial_configs == {
        0: {"name": "config_0"},
        1: {"name": "config_1"},
        2: {"name": "config_2"},
    }


"""
The test should verify that:

    the requested number of trials is executed;
    entry.sample() is called;
    Experiment.run() is called for each trial;
    y_val causes the study to maximize AUC;
    trial configurations are stored.
"""






def test_get_best_config_returns_configuration_of_best_trial():
    """Tuner should return the configuration associated with Optuna's best trial."""

    tuner = Tuner.__new__(Tuner)

    tuner.trial_configs = {
        0: {"models": {"n_estimators": 100}},
        1: {"models": {"n_estimators": 200}},
        2: {"models": {"n_estimators": 300}},
    }

    study = SimpleNamespace(
        best_trial=SimpleNamespace(
            number=1,
        )
    )

    result = tuner.get_best_config(study)

    assert result == {
        "models": {
            "n_estimators": 200,
        }
    }

"""
This is much simpler, and it's worth testing because 
it connects the Optuna result back to the actual sampled configuration.

get_best_config() is essentially a mapping:

Optuna best_trial.number
          ↓
     trial_configs
          ↓
     best config

No need to create a real Optuna study here. 
SimpleNamespace is enough because we're testing your method, not Optuna
"""




from unittest.mock import MagicMock, patch

from anomaly_detection.tuning.tuner import Tuner


def test_tuner_run_minimizes_mean_score_without_labels():
    """Tuner should minimize mean score when validation labels are unavailable."""

    fake_entry = MagicMock()

    fake_entry.sample.side_effect = [
        {"name": "config_0"},
        {"name": "config_1"},
        {"name": "config_2"},
    ]

    fake_exp = MagicMock()

    fake_exp.run.side_effect = [
        {"mean_score": 0.30},
        {"mean_score": 0.10},
        {"mean_score": 0.20},
    ]

    with patch(
        "anomaly_detection.tuning.tuner.Experiment",
        return_value=fake_exp,
    ):
        tuner = Tuner(
            model_type="isoforest",
            evaluator=MagicMock(),
            tun_cfg=MagicMock(),
            logger=MagicMock(),
        )

        with patch(
            "anomaly_detection.tuning.tuner.MODEL_REGISTRY",
            {"isoforest": lambda: fake_entry},
        ):
            study = tuner.run(
                X_train=[[1]],
                X_val=[[2]],
                y_val=None,
                n_trials=3,
            )

    assert study.direction.name == "MINIMIZE"
    assert study.best_value == 0.10

    assert fake_exp.run.call_count == 3

    assert tuner.trial_configs == {
        0: {"name": "config_0"},
        1: {"name": "config_1"},
        2: {"name": "config_2"},
    }


"""
Now the two meaningful Tuner.run() branches are covered:

    labels available → maximize auc
    no labels → minimize mean_score
    
"""