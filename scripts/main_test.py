

from anomaly_detection.evaluation.evaluator import Evaluator, ThresholdSelector
from anomaly_detection.preprocessing.pipeline import PreprocessingPipeline

from sklearn.preprocessing import StandardScaler


from anomaly_detection.experiments.experiments import Experiment


from anomaly_detection.models.factory import ModelFactory


from anomaly_detection.data.data import DataModule
from pathlib import Path
import numpy as np



BASE_DIR = Path(__file__).resolve().parents[1]  # __file__ -> actual file location
TRAIN_PATH = BASE_DIR / "data" / "servers" / "X_part2.npy"
VAL_PATH = BASE_DIR / "data" / "servers" / "X_val_part2.npy"
Y_VAL_PATH = BASE_DIR / "data" / "servers" / "y_val_part2.npy"


import hydra
from omegaconf import DictConfig




from anomaly_detection.preprocessing.transforms.scaling import ScalerTransform





@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):



    # --- 1. SETUP DATA ---
    data = DataModule(TRAIN_PATH, VAL_PATH, Y_VAL_PATH)
    X_train, X_val, y_val = data.load()

    # --- 2. SETUP COMPONENTS ---
    prep = PreprocessingPipeline([ScalerTransform(StandardScaler)])
    evaluator = Evaluator()
    selector = ThresholdSelector()
    
    # Initialize the experiment runner
    exp = Experiment(prep, selector, evaluator) #, logger=MLflowLogger())

    # --- 3. CONFIGURATION (In reality, this comes from Hydra/YAML) ---
    # Example for a Transfer Learning AE
    
    """
    train_cfg = TrainingConfig(
        strategy="transfer_learning",
        backbone_name="encoder_v1",
        phase1=PhaseConfig(lr=1e-3, epochs=10),
        phase2=PhaseConfig(lr=1e-5, epochs=5, unfreeze_layers=2)
    )
    """
    
    #model_params = {"layers": [64, 32, 16], "dropout": 0.2}
    #model_params = **cfg.model_type.models

    # --- 4. EXECUTION ---
    # Use the Factory to build the ready-to-use wrapper
    model_wrapper = ModelFactory.create(
        model_type="isoforest",
        model_cfg=cfg.model_type.models,
        runtime_params={"input_dim": X_train.shape[1]}, # later after prep i shpuild do this!!!
                                # i need to do this in teh exp
                                # bc the shape should be with X_val
        #train_cfg=train_cfg
    )
    # cleanest approach : https://chatgpt.com/c/69e2c1ad-3e74-83e9-97b2-c0f0786ca927

    # Run the experiment
    print("Starting Experiment...")
    final_metrics, final_threshold = exp.run(model_wrapper, X_train, X_val, y_val)
    
    print(f"Final F1 Score: {final_metrics['f1']:.4f}")




if __name__ == "__main__":
    main()

