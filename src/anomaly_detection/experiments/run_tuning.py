import optuna
from anomaly_detection.core.builder import AnomalyModelBuilder



class AnomalyTuner:
    def __init__(self, exp, tuning_cfg):
        self.exp = exp
        self.tuning_cfg = tuning_cfg

    def run_study(self, build_fn, X_train, X_val, y_val, n_trials=2):
        study = optuna.create_study(direction="maximize")
        
        def objective(trial):
            builder = build_fn(trial, self.tuning_cfg)
            metrics, _ = self.exp.run(builder, X_train, X_val, y_val)
            return metrics["f1"]

        study.optimize(objective, n_trials=n_trials)
        return study




#
def make_tuning_builder(model_name, model_cfg, training_cfg):
        def build_fn(trial, tuning_cfg):
            return AnomalyModelBuilder(
                model_name=model_name,
                model_cfg=model_cfg,
                training_cfg=training_cfg,
                trial=trial,
                tuning_cfg=tuning_cfg
            )

        return build_fn












