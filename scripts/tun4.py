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



    from anomaly_detection.models.ae.schemas import AETuningConfig, AETrainingTuningConfig
    from anomaly_detection.models.schemas import IntParam, FloatParam, CategoricalParam
    
    # build appr. tuning config
    def build_ae_tuning_config(cfg):
        model_tuning_cfg = AETuningConfig(
            n_layers=IntParam(**cfg.model_space.n_layers),
            encoder_dim=IntParam(**cfg.model_space.encoder_dim),
        )

        training_tuning_cfg = AETrainingTuningConfig(
            lr=FloatParam(**cfg.training_space.lr),
            batch_size=CategoricalParam(**cfg.training_space.batch_size),
            epochs=10 #cfg.training_space.epochs
        )

        return {
            "model_space": model_tuning_cfg,
            "training_space": training_tuning_cfg
        } # later not a dict but a real conf compose coinf

    # then use a registry for this; tuning_config_registry
    tuning_cfg = build_ae_tuning_config(cfg.model_type.tuning)

    #tuner = AnomalyTuner(exp, cfg.model_type.tuning)
    tuner = AnomalyTuner(exp, tuning_cfg)
    tuner.run_study(build_fn, X_train, X_val, y_val, 10)





    


if __name__ == "__main__":
    main()

# python -m scripts.tuning_testv3