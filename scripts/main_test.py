

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


#from anomaly_detection.models.factory import model_builder



@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):



    # --- 1. SETUP DATA ---
    data = DataModule(TRAIN_PATH, VAL_PATH, Y_VAL_PATH)
    X_train, X_val, y_val = data.load()

    # --- 2. SETUP COMPONENTS ---
    prep = PreprocessingPipeline([ScalerTransform(StandardScaler)])
    evaluator = Evaluator()
    selector = ThresholdSelector()


    # how to build this specific model, given runtime info
    model_builder = lambda runtime_params: ModelFactory.create(cfg.model_type.name, 
                cfg.model_type.models, runtime_params)

    
    # Initialize the experiment runner
    exp = Experiment(prep, selector, evaluator) #, logger=MLflowLogger())


    # Example for a Transfer Learning AE; i can pŕpbably handle TL via config.
    

    # Run the experiment
    print("Starting Experiment...")
    final_metrics, final_threshold = exp.run(model_builder, X_train, X_val, y_val)
    print(f"Final F1 Score: {final_metrics['f1']:.4f}")




if __name__ == "__main__":
    main()

