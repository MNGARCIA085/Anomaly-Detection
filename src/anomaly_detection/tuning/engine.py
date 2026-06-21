#This file owns the optimization mechanism (Optuna)

import optuna

class AnomalyTuner:
    def __init__(self, exp, tuning_cfg):
        self.exp = exp
        self.tuning_cfg = tuning_cfg

    def run_study(self, build_fn, X_train, X_val, y_val, n_trials=2):
        study = optuna.create_study(direction="maximize")
        
        def objective(trial):
            builder = build_fn(trial, self.tuning_cfg) # anModelBuilder, retorna modelo+trainer
            metrics, _ = self.exp.run(builder, X_train, X_val, y_val)
            return metrics["f1"]

        study.optimize(objective, n_trials=n_trials)
        return study