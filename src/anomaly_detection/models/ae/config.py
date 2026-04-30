from anomaly_detection.models.ae.schemas import AEModelTuningConfig, AETrainingTuningConfig, AETuningConfig
from anomaly_detection.models.schemas import IntParam, FloatParam, CategoricalParam

# build tuning config
def build_ae_tuning_config(cfg):
    model_tuning_cfg = AEModelTuningConfig(
        n_layers=IntParam(**cfg.model_space.n_layers),
        encoder_dim=IntParam(**cfg.model_space.encoder_dim),
    )

    training_tuning_cfg = AETrainingTuningConfig(
        lr=FloatParam(**cfg.training_space.lr),
        batch_size=CategoricalParam(**cfg.training_space.batch_size),
        epochs=10 #cfg.training_space.epochs
    )

    return AETuningConfig(model_tuning_cfg, training_tuning_cfg)





"""


models/
  ae/
    model.py        ← architecture (nn.Module, etc.)
    trainer.py      ← training logic
    builder.py      ← AEBuilder (composition logic)
    config.py       ← build_ae_tuning_config, configs
"""