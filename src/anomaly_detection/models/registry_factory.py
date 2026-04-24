


#from .aev1 import build_ae_wrapper, AEConfig, AETrainingConfig # later to AEModelConfig



# ok too
#from .aev2 import build_ae_wrapper, AEConfig, AETrainingConfig


from anomaly_detection.models.ae.wrapper import build_wrapper as build_ae_wrapper
from anomaly_detection.models.ae.schemas import AEConfig, AETrainingConfig

from anomaly_detection.models.isoforest.wrapper import build_wrapper as build_iforest_wrapper
from anomaly_detection.models.isoforest.schemas import IsoForestConfig








class ModelFactory:

    @staticmethod
    def create(name, cfg, runtime_params, trial=None):

        if name == "ae":

        	# confs from hydra, abstract later!!!
            model_cfg = AEConfig(**cfg.model_type.models) #**cfg.models.autoencoder
            training_cfg = AETrainingConfig(**cfg.model_type.training)

            
            return build_ae_wrapper(
                model_cfg,
                training_cfg,
                runtime_params, # it must include input_dim
                trial
            )

        elif name == "isoforest":

            #print(cfg.model_type.models)

            model_cfg = IsoForestConfig(**cfg.model_type.models)

            #print('cfg', model_cfg)

            return build_iforest_wrapper(
                model_cfg,
                runtime_params,
                trial
            )





def model_builder_closure(model_name, cfg, runtime_params, trial=None):
    return ModelFactory.create(
        name=model_name,
        cfg=cfg,
        runtime_params=runtime_params,
        trial=trial # Optuna trial injected here
    )




"""
class AnomalyModelBuilder:
    def __init__(self, model_name, cfg, trial=None):
        self.model_name = model_name
        self.cfg = cfg
        self.trial = trial

    def __call__(self, runtime_params):
        #This is what Experiment.run() calls.
        return ModelFactory.create(
            name=self.model_name,
            cfg=self.cfg,
            runtime_params=runtime_params,
            trial=self.trial
        )
"""






"""
WRAPPER_REGISTRY = {
    "autoencoder": build_ae_wrapper,
    #"iforest": build_iforest_wrapper,
    #"gmm": build_gmm_wrapper,
}


class ModelFactory:

    @staticmethod
    def create(name, cfg, runtime_params, trial=None):

        if name not in WRAPPER_REGISTRY:
            raise ValueError(f"Unknown model: {name}")

        builder = WRAPPER_REGISTRY[name]

        return builder(cfg[name], runtime_params, trial)
"""