from anomaly_detection.models.registry import MODEL_BUILDER
"""
bind static config now, inject runtime params later
runtime_params are only known inside Experiment.run(), so you still need AnomalyModelBuilder
"""
class AnomalyModelBuilder:
    """ returns Autoencoder wrapper model (arch+trainer)"""

    def __init__(
        self,
        model_name,
        model_cfg=None,
        training_cfg=None,
        trial=None,
        tuning_cfg=None,
    ):
        self.builder = MODEL_BUILDER[model_name]

        self.fixed_params = {
            "model_cfg": model_cfg,
            "training_cfg": training_cfg,
            "trial": trial,
            "tuning_cfg": tuning_cfg,
        }

    def __callv0__(self, runtime_params):
        """This is what Experiment.run() calls."""
        return self.builder(
            runtime_params=runtime_params,
            **self.fixed_params,
        )


    def __call__(self, runtime_params):
        """This is what Experiment.run() calls."""
        # 1. Instantiate the model wrapper
        model_wrapper = self.builder(
            runtime_params=runtime_params,
            **self.fixed_params,
        )

        # 2. Combine and attach the params directly to the model wrapper object
        all_params = {
            "model_name": self.builder.__name__,  # or pass model_name to self
            "runtime_params": runtime_params,
            **self.fixed_params,
        }

        # Dynamically add a method or attribute to the returned wrapper
        model_wrapper.get_params = lambda: all_params

        return model_wrapper



