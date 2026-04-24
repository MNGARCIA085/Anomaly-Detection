import hydra
import optuna






from anomaly_detection.evaluation.evaluator import Evaluator, ThresholdSelector
from anomaly_detection.preprocessing.pipeline import PreprocessingPipeline
from sklearn.preprocessing import StandardScaler
from anomaly_detection.experiments.experiments import Experiment

from anomaly_detection.models.registry_factory import ModelFactory

from anomaly_detection.data.data import DataModule
from pathlib import Path
import numpy as np
from omegaconf import DictConfig
from anomaly_detection.preprocessing.transforms.scaling import ScalerTransform




BASE_DIR = Path(__file__).resolve().parents[1]  # __file__ -> actual file location
TRAIN_PATH = BASE_DIR / "data" / "servers" / "X_part2.npy"
VAL_PATH = BASE_DIR / "data" / "servers" / "X_val_part2.npy"
Y_VAL_PATH = BASE_DIR / "data" / "servers" / "y_val_part2.npy"




from anomaly_detection.experiments.run_tuning import AnomalyModelBuilder




@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):

    # --- 1. DATA ---
    data = DataModule(TRAIN_PATH, VAL_PATH, Y_VAL_PATH)
    X_train, X_val, y_val = data.load()

    # --- 2. COMPONENTS ---
    prep = PreprocessingPipeline([ScalerTransform(StandardScaler)])
    evaluator = Evaluator()
    selector = ThresholdSelector()

    exp = Experiment(prep, selector, evaluator)

    # --- 3. BUILDER FACTORY (important change) ---
    
    """
    def build_wrapper_builder(trial=None):

        def wrapper_builder(runtime_params):
            return ModelFactory.create(
                cfg.model_type.name,
                #cfg.model_type.models,
                cfg,
                runtime_params,
                trial=trial
            )

        return wrapper_builder
    """



    
    print("Running single experiment...")

    #wrapper_builder = build_wrapper_builder(trial=None)

    wrapper_builder = AnomalyModelBuilder(cfg.model_type.name, cfg) # trail=None (no tuning)

    metrics, threshold = exp.run(
        wrapper_builder,
        X_train,
        X_val,
        y_val
    )

    print(f"F1: {metrics['f1']:.4f}")
    return
    


    # =========================================================
    # --- 4A. SINGLE RUN (no Optuna)
    # =========================================================
    """
    if not cfg.tuning.enabled:

        print("Running single experiment...")

        wrapper_builder = build_wrapper_builder(trial=None)

        metrics, threshold = exp.run(
            wrapper_builder,
            X_train,
            X_val,
            y_val
        )

        print(f"F1: {metrics['f1']:.4f}")
        return
    """

    # =========================================================
    # --- 4B. OPTUNA TUNING
    # =========================================================
    print("Starting Optuna tuning...")



    """
    def objective(trial):

        wrapper_builder = build_wrapper_builder(trial)

        metrics, _ = exp.run(
            wrapper_builder,
            X_train,
            X_val,
            y_val
        )

        return metrics["f1"]  # or PR-AUC

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=4) # laater form config!!!!
    """

    from anomaly_detection.experiments.run_tuning import AnomalyTuner

    tuner = AnomalyTuner(
        model_name=cfg.model_type.name, 
        cfg=cfg, 
        exp=exp
    )
    
    study = tuner.run_study(X_train, X_val, y_val, n_trials=10)
    print(f"Best Score: {study.best_value}")

    print("\nBest trial:")
    print(study.best_trial.params)
    print(f"Best F1: {study.best_value:.4f}")


if __name__ == "__main__":
    main()