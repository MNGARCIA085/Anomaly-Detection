import hydra
import optuna
from anomaly_detection.data.data import DataModule

from pathlib import Path
import numpy as np
from omegaconf import DictConfig


from anomaly_detection.new_2.experiment import Experiment


from anomaly_detection.new_2.tuner import Tuner


#------PATHS (later maybe from hydra?????)---------#
BASE_DIR = Path(__file__).resolve().parents[1]  # __file__ -> actual file location
TRAIN_PATH = BASE_DIR / "data" / "servers" / "X_part2.npy"
VAL_PATH = BASE_DIR / "data" / "servers" / "X_val_part2.npy"
Y_VAL_PATH = BASE_DIR / "data" / "servers" / "y_val_part2.npy"



@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):

    # --- 1. DATA ---
    data = DataModule(TRAIN_PATH, VAL_PATH, Y_VAL_PATH)
    X_train, X_val, y_val = data.load()

    #------------ NEW TEST---------------

    # =========================================================
    # 8. TRAIN ONLY
    # =========================================================

    def train_once(
        model_type,
        cfg,
        X_train,
        X_val
    ):

        exp = Experiment(
            model_type
        )

        return exp.run(
            cfg,
            X_train,
            X_val
        )


    # 9. TEST
    # =========================================================

    np.random.seed(0)


    """
    X_train = np.random.randn(
        200,
        20
    )

    X_val = np.random.randn(
        50,
        20
    )
    """


    # ---------- TRAIN ----------

    cfg = {

        "prep": {
            "scaler": "standard",
            "use_pca": True,
            "pca_dim": 8
        },

         "model": {
            "encoder_dims": [32, 16],
            "decoder_dims": [32],
        },

        "training": {
            "lr": 1e-3,
            "epochs": 10,
            "batch_size": 32
        }
    }

    print(
        train_once(
            "ae",
            cfg,
            X_train,
            X_val
        )
    )


    # ---------- TUNE ----------

    study = (
        Tuner("ae") # ae, iso
        .run(
            X_train,
            X_val,
            n_trials=3
        )
    )

    print(
        study.best_value
    )

    print(
        study.best_params
    )








    


if __name__ == "__main__":
    main()

# python -m scripts.tun4
#python -m scripts.tun4 model_type=isoforest
# python -m scripts.tuning_testv3


#tree -I "env|__pycache__"


"""
T DO

from hydra config
mlflow logging
better training with multiple callbacks
evaluator
real data

separate files appropiately



"""