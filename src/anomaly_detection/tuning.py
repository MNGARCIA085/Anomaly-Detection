from abc import ABC, abstractmethod

class BaseTuner(ABC):
    def __init__(self, model_factory, experiment_runner):
        self.model_factory = model_factory
        self.experiment_runner = experiment_runner

    @abstractmethod
    def tune(self, X_train, X_val, y_val, search_space: Dict):
        pass

#-----------------------------------------------------------
# SKLEARN TUNER (using GridSearchCV or RandomizedSearchCV)
#-----------------------------------------------------------
class SklearnTuner(BaseTuner):
    def tune(self, X_train, X_val, y_val, search_space):
        # Wraps the factory call into a format Sklearn understands
        # or uses the experiment_runner in a loop.
        pass

#-----------------------------------------------------------
# OPTUNA TUNER (For PyTorch/Heavy Iteration)
#-----------------------------------------------------------
import optuna

class OptunaTuner(BaseTuner):
    def __init__(self, model_factory, experiment_runner, n_trials=20):
        super().__init__(model_factory, experiment_runner)
        self.n_trials = n_trials

    def objective(self, trial, X_train, X_val, y_val, search_space):
        # 1. Suggest hyperparams from the search space
        sampled_cfg = {}
        for key, space in search_space.items():
            if space['type'] == 'categorical':
                sampled_cfg[key] = trial.suggest_categorical(key, space['choices'])
            elif space['type'] == 'float':
                sampled_cfg[key] = trial.suggest_float(key, space['low'], space['high'], log=space.get('log', False))
            elif space['type'] == 'int':
                sampled_cfg[key] = trial.suggest_int(key, space['low'], space['high'])

        # 2. Build model using existing Factory
        # This ensures the model built for tuning is identical to the one in production
        model = self.model_factory.create_anomaly_model(
            model_type=sampled_cfg['model_type'],
            model_params=sampled_cfg['model_params'],
            train_cfg=sampled_cfg['train_cfg']
        )

        # 3. Run a standard experiment
        metrics, _ = self.experiment_runner.run_with_model(model, X_train, X_val, y_val)
        
        # 4. Return the metric to optimize (e.g., F1 Score)
        return metrics['f1']

    def tune(self, X_train, X_val, y_val, search_space):
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda t: self.objective(t, X_train, X_val, y_val, search_space), n_trials=self.n_trials)
        return study.best_params



"""
for pruning
# Inside OptunaTuner.objective:
# If the intermediate loss is bad, stop the trial early
trial.report(current_loss, step=epoch)
if trial.should_prune():
    raise optuna.exceptions.TrialPruned()
"""
