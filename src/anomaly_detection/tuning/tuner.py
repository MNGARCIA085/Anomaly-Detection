


from anomaly_detection.experiments.experiments import Experiment

from anomaly_detection.models.registry import MODEL_REGISTRY


import optuna






class Tuner:

    def __init__(
        self,
        model_type,
        evaluator,
        # config
        tun_cfg,
    ):

        self.model_type = (
            model_type
        )

        self.exp = (
            Experiment(
                model_type,
                evaluator
            )
        )


        # new
        self.tun_cfg = tun_cfg


    def run(
        self,
        X_train,
        X_val,
        y_val=None,
        n_trials=5
    ):

        entry = (
            MODEL_REGISTRY[
                self.model_type
            ]
        )



        direction = (
            "maximize"
            if y_val is not None
            else "minimize"
        )

        def objective(
            trial
        ):


            cfg = (
                entry.sample(
                    trial,
                    self.tun_cfg,
                )
            )



            result = (
                self.exp.run(
                    cfg['prep'],
                    cfg['model'],
                    cfg['training'],
                    X_train,
                    X_val,
                    y_val
                )
            )

            if y_val is not None:
                return (
                    result["auc"]
                )

            return (
                result[
                    "mean_score"
                ]
            )

        study = (
            optuna.create_study(
                direction=direction
            )
        )

        study.optimize(
            objective,
            n_trials=n_trials
        )

        return study











#-----------------------------
class Tunerv0:

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