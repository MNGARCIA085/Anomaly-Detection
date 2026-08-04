from anomaly_detection.models.registry import register
from .schemas import AEConfig
from anomaly_detection.training.schemas import TrainingConfig

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

from anomaly_detection.preprocessing.pipeline import PreprocessingPipeline



from .model import AE, AEWrapper


from anomaly_detection.training.callbacks import EarlyStopping,PrintLossCallback


from anomaly_detection.training.registry import TRAINER_REGISTRY


from ...base_entry import BaseModelEntry



#from anomaly_detection.training.optimizers import create_optimizer
from anomaly_detection.training.losses import create_loss


from anomaly_detection.training.optimizers import sample_optimizer, create_optimizer



@register("ae")
class AEEntry(BaseModelEntry):
    

    # sample only for tuning; maybe later more generic
    def sample(self, trial, tun_cfg):

        return {

            "name": "ae",

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


            "models": {
                "encoder_dims": [
                    trial.suggest_int("enc1", 16, 64),
                    trial.suggest_int("enc2", 4, 32),
                ],

                "decoder_dims": [
                    trial.suggest_int("dec1", 16, 64)
                ]
            },


            "training": {
                
                "optimizer": sample_optimizer(
                    trial,
                    tun_cfg.training_space.optimizer
                ),

                "loss": {
                    "name": "mse"
                },

                "epochs": trial.suggest_int(
                    "epochs",
                    tun_cfg.training_space.epochs.low,
                    tun_cfg.training_space.epochs.high
                ),

                "batch_size": trial.suggest_categorical(
                    "batch_size",
                    tun_cfg.training_space.batch_size.choices
                ),

                "type": "default"

            }

        }


    def build_preprocessor(self, prep_cfg): # later -> prep_cfg:AEPrepConfig or like that

        steps = []

        if prep_cfg["scaler"] == "standard":
            steps.append(StandardScaler())

        elif prep_cfg["scaler"] == "minmax":
            steps.append(MinMaxScaler())

        """
        if prep_cfg["use_pca"]:

            steps.append(
                PCA(
                    n_components=prep_cfg["pca_dim"]
                )
            )
        """

        return PreprocessingPipeline(
            steps
        )



    def build(
        self,
        cfg_model,
        cfg_training,
        input_dim
    ):

        print(type(cfg_training))
        print(cfg_training)
        print(cfg_training["optimizer"])

        # later maybe move it out
        model_cfg = AEConfig(
            input_dim=input_dim,
            encoder_dims=cfg_model["encoder_dims"],
            decoder_dims=cfg_model["decoder_dims"],
        )

        model = AE(model_cfg)


        # trainer
        optimizer = create_optimizer(
            cfg_training["optimizer"],
            model.parameters()
        )


        print('dsfdsfdssfdsdsf')

        loss = create_loss(
            cfg_training["loss"],
        )


        trainer_cfg = TrainingConfig(
            epochs=cfg_training["epochs"],
            batch_size=cfg_training["batch_size"],
            optimizer=optimizer,
            loss=loss,
            callbacks=[
                EarlyStopping(patience=3),
                PrintLossCallback(),
            ]
        )


        
        trainer_cls = TRAINER_REGISTRY[cfg_training["type"]]
        trainer = trainer_cls(trainer_cfg)


        return AEWrapper(
            model,
            trainer
        )


    # load model (to simplify inference pipeline)
    def load(self, path):

        return AEWrapper.load(path)











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