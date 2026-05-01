from anomaly_detection.models.ae.schemas import AEModelTuningConfig, AETrainingTuningConfig, AETuningConfig
from anomaly_detection.models.schemas import IntParam, FloatParam, CategoricalParam




from anomaly_detection.models.ae.trainer import PrintLossCallback, EarlyStopping


# build tuning config
def build_ae_tuning_config(cfg):
    model_tuning_cfg = AEModelTuningConfig(
        n_layers=IntParam(**cfg.model_space.n_layers),
        encoder_dim=IntParam(**cfg.model_space.encoder_dim),
    )

    training_tuning_cfg = AETrainingTuningConfig(
        lr=FloatParam(**cfg.training_space.lr),
        batch_size=CategoricalParam(**cfg.training_space.batch_size),
        epochs=10, #cfg.training_space.epochs

        # hardcoded for now!!!!
        callbacks=[PrintLossCallback(), EarlyStopping(patience=3)]
        #callbacks=[]

    )

    return AETuningConfig(model_tuning_cfg, training_tuning_cfg)



#------------------------for training only------------------


from anomaly_detection.models.ae.schemas import AETrainingConfig


# hardcoded for quick tests
def build_ae_training_config(cfg):
    return AETrainingConfig(
        lr=1e-3,  #(**cfg.training.lr),
        batch_size=32, # (**cfg.training.batch_size),
        epochs=10, #cfg.training_space.epochs
        # hardcoded for now!!!!
        callbacks=[PrintLossCallback(), EarlyStopping(patience=2)]
        #callbacks=[]

    )






"""


models/
  ae/
    model.py        ← architecture (nn.Module, etc.)
    trainer.py      ← training logic
    builder.py      ← AEBuilder (composition logic)
    config.py       ← build_ae_tuning_config, configs
"""