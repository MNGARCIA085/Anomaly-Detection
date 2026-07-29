import hydra
import optuna
from anomaly_detection.data.data import DataModule

from pathlib import Path
import numpy as np
from omegaconf import DictConfig


from anomaly_detection.experiments.experiments import Experiment


from anomaly_detection.tuning.tuner import Tuner




from anomaly_detection.evaluation.evaluator import Evaluator




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

    

    model_type = cfg.model_type.name
    print(model_type)


    # =========================================================
    # 8. TRAIN ONLY
    # =========================================================



    def train_once(
        model_type,
        prep_cfg,
        model_cfg,
        train_cfg,
        X_train,
        X_val,
        y_val,
    ):


        exp = Experiment(
            model_type=model_type,
            evaluator=Evaluator()
        )


        return exp.run(
            prep_cfg,
            model_cfg,
            train_cfg,
            X_train,
            X_val,
            y_val
        )


    print(
        train_once(
            model_type, # ae, iso
            cfg.model_type.prep,
            cfg.model_type.models,
            cfg.model_type.training,
            X_train,
            X_val,
            y_val
        )
    )





    #===========TUNING===============

    tun_cfg = cfg.model_type.tuning

    print(tun_cfg)


    tuner = Tuner(
        model_type, # model_type; ae
        Evaluator(), #evaluator=Evaluator(),
        tun_cfg
    )

    study = tuner.run(
        X_train,
        X_val,
        y_val,
        n_trials=2
    )

    print(
        study.best_value
    )

    print(
        study.best_params
    )




    return





if __name__ == "__main__":
    main()




# python -m scripts.main
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