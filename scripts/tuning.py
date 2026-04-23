


class Tuner:
    def __init__(self, experiment, direction="maximize"):
        self.experiment = experiment
        self.direction = direction

    def optimize(self, build_model_fn, X_train, X_val, y_val, n_trials):

        def objective(trial):
            model_builder = build_model_fn(trial)

            metrics, _ = self.experiment.run(
                model_builder,
                X_train,
                X_val,
                y_val
            )

            return metrics["f1"]  # or pr_auc

        study = optuna.create_study(direction=self.direction)
        study.optimize(objective, n_trials=n_trials)

        return study



# exactly ius a warpper
def build_model_fn(cfg):
    
    def builder(trial):

        def model_builder(runtime_params): # exactly is wrapper_builder!!!!!
            return ModelFactory.create(
                cfg.model_type.name,
                cfg.model_type.models,
                runtime_params,
                trial=trial   # 👈 inject trial here
            )

        return model_builder

    return builder





 class ModelFactory:

    @staticmethod
    def create(name, cfg, runtime_params, trial=None):

        if name == "autoencoder":
            hidden_dim = trial.suggest_int("hidden_dim", 16, 128)
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

            model = AE(input_dim=runtime_params["input_dim"],
                       hidden_dim=hidden_dim)

            trainer = AETrainer(lr=lr)

            return AutoencoderModel(model, trainer)

        elif name == "iforest":
            n_estimators = trial.suggest_int("n_estimators", 50, 300)

            model = IsolationForest(n_estimators=n_estimators)

            return SklearnWrapper(model)



exp = Experiment(prep, selector, evaluator)

tuner = Tuner(exp)

study = tuner.optimize(
    build_model_fn(cfg),
    X_train,
    X_val,
    y_val,
    n_trials=50
)

print("Best trial:", study.best_trial.params)




"""
remeber threshold stuff

-> that optuna sets it: threshold = trial.suggest_float("threshold", min_score, max_score)

"""