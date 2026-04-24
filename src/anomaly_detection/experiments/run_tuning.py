import optuna
from anomaly_detection.models.registry_factory import ModelFactory






class AnomalyModelBuilder:
    def __init__(self, model_name, cfg, trial=None):
        self.model_name = model_name
        self.cfg = cfg
        self.trial = trial

    def __call__(self, runtime_params):
        """This is what Experiment.run() calls."""
        return ModelFactory.create(
            name=self.model_name,
            cfg=self.cfg,
            runtime_params=runtime_params,
            trial=self.trial
        )






class AnomalyTuner:
    def __init__(self, model_name, cfg, exp):
        self.model_name = model_name
        self.cfg = cfg
        self.exp = exp

    def run_study(self, X_train, X_val, y_val, n_trials=20):
        study = optuna.create_study(direction="maximize")
        
        def objective(trial):
            # We create the builder with the trial and pass it to exp
            builder = AnomalyModelBuilder(self.model_name, self.cfg, trial=trial)
            
            metrics, _ = self.exp.run(builder, X_train, X_val, y_val)
            return metrics["f1"]

        study.optimize(objective, n_trials=n_trials)
        return study













