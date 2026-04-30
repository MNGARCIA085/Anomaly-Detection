import optuna
from anomaly_detection.models.registry_factory import ModelFactory






class AnomalyModelBuilder:
    def __init__(self, model_name, model_cfg=None, training_cfg=None, trial=None, tuning_cfg=None):
        self.model_name = model_name
        self.model_cfg = model_cfg
        self.training_cfg = training_cfg
        self.trial = trial
        self.tuning_cfg = tuning_cfg
        

    def __call__(self, runtime_params):
        """This is what Experiment.run() calls."""
        return ModelFactory.create(
            name=self.model_name,
            model_cfg=self.model_cfg,
            runtime_params=runtime_params,
            training_cfg=self.training_cfg,
            trial=self.trial,
            tuning_cfg=self.tuning_cfg
        )





class AnomalyTuner:
    def __init__(self, exp, tuning_cfg):
        self.exp = exp
        self.tuning_cfg = tuning_cfg

    def run_study(self, build_fn, X_train, X_val, y_val, n_trials=20):
        study = optuna.create_study(direction="maximize")
        
        def objective(trial):
            builder = build_fn(trial, self.tuning_cfg)
            metrics, _ = self.exp.run(builder, X_train, X_val, y_val)
            return metrics["f1"]

        study.optimize(objective, n_trials=n_trials)
        return study

















