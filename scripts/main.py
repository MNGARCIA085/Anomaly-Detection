import hydra
import optuna
from anomaly_detection.data.data import DataModule

from pathlib import Path
import numpy as np
from omegaconf import DictConfig

from anomaly_detection.experiments.experiments import Experiment
from anomaly_detection.tuning.tuner import Tuner
from anomaly_detection.evaluation.evaluator import Evaluator

from hydra.utils import to_absolute_path



@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):



    # --- 1. DATA ---
    # separate from exp. bc of loading in tuning    
    data = DataModule(
        to_absolute_path(cfg.data.train_path),
        to_absolute_path(cfg.data.val_path),
        to_absolute_path(cfg.data.y_val_path),
    )
    X_train, X_val, y_val = data.load()
    

    model_type = cfg.model_type.name
    print(model_type)


    # =========================================================
    # 8. TRAIN ONLY
    # =========================================================


    from anomaly_detection.infra.logging.mlflow_logger import  MLFlowLogger


    
    def train_once(
        model_type,
        cfg,
        all_cfg,
        X_train,
        X_val,
        y_val,
    ):


        exp = Experiment(
            model_type=model_type,
            evaluator=Evaluator(),
            logger=MLFlowLogger(
                tracking_db=to_absolute_path(all_cfg.paths.mlflow_db),
                artifact_dir=to_absolute_path(all_cfg.paths.mlflow_artifacts),
            ),
        )


        return exp.run(
            cfg,
            X_train,
            X_val,
            y_val
        )



    
    print(
        train_once(
            model_type, # ae, iso
            cfg.model_type,
            cfg,
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
        tun_cfg,
        MLFlowLogger(),
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



    # retrain best model
    # ---------------------------------
    # FINAL TRAINING
    # ---------------------------------

    best_cfg = tuner.get_best_config(study)

    print("Best config:")
    print(best_cfg)

    final_model = train_once(
        model_type,
        best_cfg,
        cfg,
        X_train,
        X_val,
        y_val
    )

    print(final_model)




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