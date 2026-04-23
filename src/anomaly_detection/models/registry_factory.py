


#from .aev1 import build_ae_wrapper, AEConfig, AETrainingConfig # later to AEModelConfig
from .aev2 import build_ae_wrapper, AEConfig, AETrainingConfig




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

        elif name == "iforest":

            model_cfg = IsoForestConfig(**cfg.models.iforest)

            return build_iforest_wrapper(
                model_cfg,
                runtime_params,
                trial
            )



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