


from anomaly_detection.new.experiment import Experiment

from anomaly_detection.new.registry import MODEL_REGISTRY


import optuna


class Tuner:

    def __init__(
        self,
        model_type
    ):

        self.model_type = model_type

        self.exp = (
            Experiment(
                model_type
            )
        )

    def run(
        self,
        X_train,
        X_val,
        n_trials=5
    ):

        entry = MODEL_REGISTRY[
            self.model_type
        ]

        def objective(trial):

            cfg = entry.sample(
                trial
            )

            return (
                self.exp.run(
                    cfg,
                    X_train,
                    X_val
                )["score"]
            )

        study = (
            optuna.create_study(
                direction="maximize"
            )
        )

        study.optimize(
            objective,
            n_trials=n_trials
        )

        return study