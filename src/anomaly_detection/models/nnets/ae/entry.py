


from anomaly_detection.models.registry import register
from .schemas import AEConfig
from anomaly_detection.training.schemas import TrainingConfig
from anomaly_detection.preprocessing.pipeline import PreprocessingPipeline
from .model import AE, AEWrapper
from anomaly_detection.training.registry import TRAINER_REGISTRY
from ...base_entry import BaseModelEntry
from anomaly_detection.training.losses import create_loss
from anomaly_detection.training.optimizers import sample_optimizer, create_optimizer
from anomaly_detection.preprocessing.components.scalers import create_scaler, sample_scaler
from anomaly_detection.preprocessing.components.transforms import create_transform
from anomaly_detection.preprocessing.components.transforms import sample_transform
from anomaly_detection.preprocessing.components.imputation import create_imputer
from anomaly_detection.preprocessing.components.imputation import sample_imputation


from anomaly_detection.training.callbacks.registry import create_callbacks
from anomaly_detection.tuning.sample_training import sample_callbacks






#from anomaly_detection.training.callbacks import EarlyStopping,PrintLossCallback



@register("ae")
class AEEntry(BaseModelEntry):
    

    # sample only for tuning; maybe later more generic
    def sample(self, trial, tun_cfg):
        """ Note. Model and threshold with hardcoded values for now (not from config)"""

        return {

            "name": "ae",


            "prep": {
                "imputation": sample_imputation(
                        trial,
                        tun_cfg.prep.imputation,
                    ),
                "transform": sample_transform(
                    trial,
                    tun_cfg.prep.transform,
                ),
                "scaler": sample_scaler(
                    trial,
                    tun_cfg.prep.scaler,
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

                "callbacks": sample_callbacks(
                    trial,
                    tun_cfg.training_space.callbacks,
                ),

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

            },

            
            "thresholding": {
                "name": "quantile",
                "params": {
                    "quantile": trial.suggest_float(
                        "threshold_quantile",
                        0.95,
                        0.999,
                    )
                },
            },
            

        }


    def build_preprocessor(self, prep_cfg): # later -> prep_cfg:AEPrepConfig or like that

        steps = []

        # imputation
        imputer_cfg = prep_cfg["imputation"]

        if imputer_cfg["enabled"]:
            imputer = create_imputer(
                imputer_cfg["name"],
                **imputer_cfg.get("params", {}),
            )

            steps.append(imputer)

        # transform
        transform_cfg = prep_cfg["transform"] # get, None

        if transform_cfg["enabled"]:
            transform = create_transform(
                transform_cfg["name"]
            )
            steps.append(transform)


        # scaler
        scaler_cfg = prep_cfg["scaler"]

        scaler = create_scaler(
            scaler_cfg["name"],
            **scaler_cfg.get("params", {}),
        )

        steps.append(scaler)


        return PreprocessingPipeline(
            steps
        )


    # new
    def adapt_input(self, X):
        # should be (N, T*F)
        return X.reshape(X.shape[0], -1)



    def build(
        self,
        cfg_model,
        cfg_training,
        input_shape,
    ):

        input_dim = input_shape[1]

        # later maybe move it out
        model_cfg = AEConfig(
            input_dim=input_dim,
            encoder_dims=cfg_model["encoder_dims"],
            decoder_dims=cfg_model["decoder_dims"],
        )

        model = AE(model_cfg)


        # optimizer
        optimizer = create_optimizer(
            cfg_training["optimizer"],
            model.parameters()
        )

        # loss
        loss = create_loss(
            cfg_training["loss"],
        )


        # create callbacks
        callbacks = create_callbacks(cfg_training["callbacks"])
        

        # trainer
        trainer_cfg = TrainingConfig(
            epochs=cfg_training["epochs"],
            batch_size=cfg_training["batch_size"],
            optimizer=optimizer,
            loss=loss,
            callbacks=callbacks
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