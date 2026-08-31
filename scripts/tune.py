import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from anomaly_detection.data.data import DataModule
from anomaly_detection.experiments.experiments import Experiment
from anomaly_detection.evaluation.evaluator import Evaluator
from anomaly_detection.infra.logging.mlflow_logger import  MLFlowLogger
from anomaly_detection.tuning.tuner import Tuner


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):


    # =========== DATA ============= #
    # separate from exp. bc of loading in tuning    
    data = DataModule(
        to_absolute_path(cfg.data.train_path),
        to_absolute_path(cfg.data.val_path),
        to_absolute_path(cfg.data.y_val_path),
    )
    X_train, X_val, y_val = data.load()
    

    model_type = cfg.model_type.name
    print(model_type)


    #===========TUNING=============== #
    tuner = Tuner(
        model_type,
        Evaluator(), 
        cfg.model_type.tuning,
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



    # ========= Retrain best model ========= #
    best_cfg = tuner.get_best_config(study)

    print("Best config:")
    print(best_cfg)


    exp = Experiment(
        model_type=model_type,
        evaluator=Evaluator(),
        logger=MLFlowLogger(
            tracking_db=to_absolute_path(cfg.paths.mlflow_db),
            artifact_dir=to_absolute_path(cfg.paths.mlflow_artifacts),
        ),
    )


    metrics = exp.run(
        best_cfg,
        X_train,
        X_val,
        y_val
    )



    print(metrics)   




    return





if __name__ == "__main__":
    main()



