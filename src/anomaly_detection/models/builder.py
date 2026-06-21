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

    def __call__(self, runtime_params):
        """This is what Experiment.run() calls."""
        return self.builder(
            runtime_params=runtime_params,
            **self.fixed_params,
        )