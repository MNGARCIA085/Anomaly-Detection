

"""
bind static config now, inject runtime params later


runtime_params are only known inside Experiment.run(), so you still need AnomalyModelBuilder

"""

from anomaly_detection.models.registry import MODEL_BUILDER

class AnomalyModelBuilder:

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

        return self.builder(
            runtime_params=runtime_params,
            **self.fixed_params,
        )






from anomaly_detection.models.factory import ModelFactory
class AnomalyModelBuilderv0:
    def __init__(self, model_name, model_cfg=None, training_cfg=None, trial=None, tuning_cfg=None):
        self.model_name = model_name
        self.model_cfg = model_cfg
        self.training_cfg = training_cfg
        self.trial = trial
        self.tuning_cfg = tuning_cfg
        

    def __call__(self, runtime_params):
        """This is what Experiment.run() calls."""
        return ModelFactory.create(
            name=self.model_name,
            model_cfg=self.model_cfg,
            runtime_params=runtime_params,
            training_cfg=self.training_cfg,
            trial=self.trial,
            tuning_cfg=self.tuning_cfg
        )