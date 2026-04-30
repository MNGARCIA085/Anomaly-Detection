from anomaly_detection.models.ae.wrapper import build_wrapper as build_ae_wrapper
from anomaly_detection.models.ae.schemas import AEConfig, AETrainingConfig

from anomaly_detection.models.isoforest.wrapper import build_wrapper as build_iforest_wrapper
from anomaly_detection.models.isoforest.schemas import IsoForestConfig


class ModelFactory:

    @staticmethod
    def create(name, model_cfg, runtime_params, training_cfg=None, trial=None, tuning_cfg=None):
        if name == "ae":
            # manages std training and also tuning            
            return build_ae_wrapper(
                model_cfg,
                training_cfg,
                runtime_params, # it must include input_dim
                trial,
                tuning_cfg
            )

        elif name == "isoforest":
            return build_iforest_wrapper(
                model_cfg,
                runtime_params,
                trial, 
                tuning_cfg
            )









