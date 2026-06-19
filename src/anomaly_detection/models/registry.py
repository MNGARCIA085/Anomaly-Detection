#all registries


from anomaly_detection.models.ae.config import build_ae_training_config, build_ae_tuning_config
from anomaly_detection.models.isoforest.config import build_iso_training_config, build_iso_tuning_config

"""
MODEL_REGISTRY = {
    "ae": AEBuilder,
    "isoforest": IFBuilder,
}
"""


TRAINER_REGISTRY = {
    #"ae_standard": AETrainer,
    "ae": build_ae_training_config,
    "isoforest": build_iso_training_config
}


TUNING_CONFIG_REGISTRY = {
    "ae": build_ae_tuning_config,
    "isoforest": build_iso_tuning_config,
}













