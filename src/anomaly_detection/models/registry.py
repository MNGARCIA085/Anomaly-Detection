#all registries


from anomaly_detection.models.ae.config import build_ae_tuning_config


"""
MODEL_REGISTRY = {
    "ae": AEBuilder,
    "isoforest": IFBuilder,
}
"""

TUNING_CONFIG_REGISTRY = {
    "ae": build_ae_tuning_config,
    #"isoforest": build_if_tuning_config,
}




"""
# if i decpuple later

TRAINER_REGISTRY = {
    "ae_standard": AETrainer,
    ...
}
"""