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




#from anomaly_detection.experiments.run_tuning import AnomalyModelBuilder




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
    
    from anomaly_detection.experiments.run_tuning import AnomalyModelBuilder, AnomalyTuner


    def make_build_fn(model_name, model_cfg, training_cfg):
        def build_fn(trial, tuning_cfg):
            return AnomalyModelBuilder(
                model_name=model_name,
                model_cfg=model_cfg,
                training_cfg=training_cfg,
                trial=trial,
                tuning_cfg=tuning_cfg
            )

        return build_fn


    
    build_fn = make_build_fn(
        model_name=cfg.model_type.name,
        model_cfg=cfg.model_type.models,
        training_cfg=cfg.model_type.training
    )




    # single training!!!!
    print(cfg.training)
    from anomaly_detection.models.ae.config import build_ae_training_config

    training_cfg = build_ae_training_config(cfg)


    wrapper_builder = AnomalyModelBuilder(cfg.model_type.name, 
                        cfg.model_type.models, training_cfg) # trail=None (no tuning)

    metrics, threshold = exp.run(
        wrapper_builder,
        X_train,
        X_val,
        y_val
    )

    print(metrics)


    return

    



    #--------TUNING


    #from anomaly_detection.models.ae.config import build_ae_tuning_config

    from anomaly_detection.models.registry import TUNING_CONFIG_REGISTRY
    build_cfg_fn = TUNING_CONFIG_REGISTRY[cfg.model_type.name]

    tuning_cfg = build_cfg_fn(cfg.model_type.tuning)


    
    tuner = AnomalyTuner(exp, tuning_cfg)
    tuner.run_study(build_fn, X_train, X_val, y_val, 10)


    """
    # then use a registry for this; tuning_config_registry
    tuning_cfg = build_ae_tuning_config(cfg.model_type.tuning)

    #tuner = AnomalyTuner(exp, cfg.model_type.tuning)
    tuner = AnomalyTuner(exp, tuning_cfg)
    tuner.run_study(build_fn, X_train, X_val, y_val, 10)
    """





    


if __name__ == "__main__":
    main()

# python -m scripts.tun4
# python -m scripts.tuning_testv3