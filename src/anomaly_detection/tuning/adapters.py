#This is a structural factory/adapter function that bridges your model builder with Optuna.
from anomaly_detection.models.builder import AnomalyModelBuilder

# build_fn is mostly an adapter that exists to inject trial and tuning_cfg.
def make_tuning_builder(model_name, model_cfg, training_cfg):
        def build_fn(trial, tuning_cfg):
            return AnomalyModelBuilder(
                model_name=model_name,
                model_cfg=model_cfg,
                training_cfg=training_cfg,
                trial=trial,
                tuning_cfg=tuning_cfg
            )

        return build_fn