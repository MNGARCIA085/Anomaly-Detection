from anomaly_detection.experiments.experiments import Experiment
from anomaly_detection.models.registry import MODEL_REGISTRY
import optuna


class Tuner:

    def __init__(
        self,
        model_type,
        evaluator,
        tun_cfg,
        logger,
    ):
        self.model_type = model_type

        self.exp = Experiment(
            model_type,
            evaluator,
            logger,
        )

        self.tun_cfg = tun_cfg
        self.trial_configs = {}


    def run(
        self,
        X_train,
        X_val,
        y_val=None,
        n_trials=5
    ):

        entry = MODEL_REGISTRY[
            self.model_type
        ]()

        direction = (
            "maximize"
            if y_val is not None
            else "minimize"
        )

        def objective(trial):

            cfg = entry.sample(
                trial,
                self.tun_cfg,
            )

            # Keep the config used by this trial
            self.trial_configs[
                trial.number
            ] = cfg

            result = self.exp.run(
                cfg,
                X_train,
                X_val,
                y_val,
                "tuning"
            )

            if y_val is not None:
                return result["auc"]

            return result["mean_score"]


        study = optuna.create_study(
            direction=direction
        )

        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=4,
        )

        return study



    def get_best_config(self, study):

        return self.trial_configs[
            study.best_trial.number
        ]



"""
One thing I would change later, though: trial_configs being an in-memory dictionary i
s fine for your current quick experiment, but if you want your architecture to be robust, I'd eventually make the best config part of the Tuner r
esult rather than exposing trial_configs directly. For now, I would keep it this simple.
"""


"""
safety
def objective(self, trial):

    cfg = self.entry.sample(
        trial,
        self.tun_cfg,
    )

    try:

        result = self.exp.run(
            cfg=cfg,
            X_train=self.X_train,
            X_val=self.X_val,
            y_val=self.y_val,
            run_type="tune",
        )

        return result["auc"]

    except Exception as e:

        print(
            f"Trial {trial.number} failed"
        )

        print(f"Error: {e}")

        print("Config:")
        print(cfg)

        raise



Bad hyperparameters
        ↓
Training becomes numerically unstable
        ↓
Raise meaningful exception
        ↓
Trial marked as failed
        ↓
Other Optuna trials continue

So yes: log what happened, but don't silently swallow the exception. Let Optuna 
know the trial failed. Later, if you want, 
you can explicitly configure the study to continue for expected numerical errors.



maybe detetct it before
if not np.isfinite(scores).all():
    raise ValueError(
        "Non-finite anomaly scores detected"
    )

"""


