from anomaly_detection.models.ae.schemas import AETrainingConfig, AEModelTuningConfig, AETrainingTuningConfig, AETuningConfig
from anomaly_detection.core.schemas import IntParam, FloatParam, CategoricalParam
from anomaly_detection.models.ae.trainer import PrintLossCallback, EarlyStopping


#------------------------for training only------------------

# hardcoded for quick tests
def build_ae_training_config(cfg): # ill pass it training_cfg
    return AETrainingConfig(
        lr=cfg.lr,
        batch_size=cfg.batch_size, 
        epochs=cfg.epochs,
        callbacks=[PrintLossCallback(), EarlyStopping(patience=2)]
    )


# build tuning config
def build_ae_tuning_config(cfg): # make instead of build????????; create better
    model_tuning_cfg = AEModelTuningConfig(
        n_layers=IntParam(**cfg.model_space.n_layers),
        encoder_dim=IntParam(**cfg.model_space.encoder_dim),
    )

    training_tuning_cfg = AETrainingTuningConfig(
        lr=FloatParam(**cfg.training_space.lr),
        batch_size=CategoricalParam(**cfg.training_space.batch_size),
        epochs=10, #cfg.training_space.epochs
        callbacks=[PrintLossCallback(), EarlyStopping(patience=3)]
    )

    return AETuningConfig(model_tuning_cfg, training_tuning_cfg)










