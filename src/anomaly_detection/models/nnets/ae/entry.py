from anomaly_detection.models.registry import register
from .schemas import AEConfig, AETrainingConfig


from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

from anomaly_detection.preprocessing.pipeline import PreprocessingPipeline



from .model import AE, AEWrapper


from anomaly_detection.models.nnets.training.callbacks import EarlyStopping,PrintLossCallback




from anomaly_detection.models.nnets.training.trainer import BaseTrainer



@register("ae")
class AEEntry:
    

    # sample only for tuning
    @staticmethod
    def sample(trial, tun_cfg):

        return {

            "prep": {
                "scaler": trial.suggest_categorical(
                    "scaler",
                    ["standard", "minmax"]
                ),

                "use_pca": trial.suggest_categorical(
                    "use_pca",
                    [True, False]
                ),

                "pca_dim": trial.suggest_int(
                    "pca_dim",
                    2,
                    10
                ),
            },


            "model": {
                "encoder_dims": [
                    trial.suggest_int("enc1", 16, 64),
                    trial.suggest_int("enc2", 4, 32),
                ],

                "decoder_dims": [
                    trial.suggest_int("dec1", 16, 64)
                ]
            },


            "training": {

                "lr": trial.suggest_float(
                    "lr",
                    tun_cfg.training_space.lr.low,
                    tun_cfg.training_space.lr.high,
                    log=True
                ),

                "epochs": trial.suggest_int(
                    "epochs",
                    tun_cfg.training_space.epochs.low,
                    tun_cfg.training_space.epochs.high
                ),

                "batch_size": trial.suggest_categorical(
                    "batch_size",
                    tun_cfg.training_space.batch_size.choices
                )
            }

        }

    @staticmethod
    def build_preprocessor(cfg):

        steps = []

        if cfg["prep"]["scaler"] == "standard":
            steps.append(StandardScaler())

        elif cfg["prep"]["scaler"] == "minmax":
            steps.append(MinMaxScaler())

        if cfg["prep"]["use_pca"]:

            steps.append(
                PCA(
                    n_components=cfg["prep"]["pca_dim"]
                )
            )

        return PreprocessingPipeline(
            steps
        )

    @staticmethod
    def build(
        cfg,
        input_dim
    ):

        model_cfg = AEConfig(
            input_dim=input_dim,
            encoder_dims=cfg["model"]["encoder_dims"], # uso lo que paso en config!!!
            decoder_dims=cfg["model"]["decoder_dims"],
        )

        model = AE(model_cfg)


        trainer_cfg = AETrainingConfig(
                lr=cfg["training"]["lr"],
                epochs=cfg["training"]["epochs"],
                batch_size=cfg["training"]["batch_size"],
                callbacks=[
                    EarlyStopping(patience=3),
                    PrintLossCallback(),
                ]
            )


        


        trainer = BaseTrainer(trainer_cfg)

        print(trainer)

        return AEWrapper(
            model,
            trainer
        )






"""
    Model entry responsible for assembling all Autoencoder-specific components.

    Acts as the integration point between the experiment framework and the AE
    implementation by encapsulating:

    - hyperparameter search space definition (`sample`)
    - preprocessing construction (`build_preprocessor`)
    - model and training assembly (`build`)

    This class allows the experiment/tuning pipeline to remain model-agnostic:
    callers interact with a common entry interface without knowing how the AE
    is configured internally.

    Responsibilities:
        - Define tunable preprocessing, model, and training parameters
        - Build the preprocessing pipeline for AE workflows
        - Construct and return a fully configured AE wrapper

    Does NOT:
        - execute training
        - run evaluation
        - perform logging
        - orchestrate experiments

    Expected interface:
        sample(trial) -> dict
        build_preprocessor(cfg) -> PreprocessingPipeline
        build(cfg, input_dim) -> ModelWrapper
    """