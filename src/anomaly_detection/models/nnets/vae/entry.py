
#https://chatgpt.com/c/6a74e018-c794-83e9-a42a-99b566358b62


from anomaly_detection.models.registry import register
from .schemas import VAEConfig
from anomaly_detection.training.schemas import TrainingConfig
from anomaly_detection.preprocessing.pipeline import PreprocessingPipeline
from .model import VAE, VAEWrapper
from anomaly_detection.training.registry import TRAINER_REGISTRY
from ...base_entry import BaseModelEntry
from anomaly_detection.preprocessing.components.scalers import create_scaler


from anomaly_detection.training.callbacks.registry import create_callbacks
from anomaly_detection.training.optimizers.registry import create_optimizer
from anomaly_detection.tuning.sample_training import sample_callbacks, sample_optimizer


# no loss object, since the VAE computes reconstruction + KL internally.


from anomaly_detection.tuning.sample_prep import(
        sample_window_size, sample_scaler
    )



@register("vae")
class VAEEntry(BaseModelEntry):

    def sample(self, trial, tun_cfg):

        return {

            "name": "vae",

            "data": {
                "windowing": {
                    "size": sample_window_size(
                        trial,
                        tun_cfg.data.windowing.size,
                    ),
                },
            },

            "prep": {
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

                "latent_dim": trial.suggest_int(
                    "latent_dim",
                    2,
                    16
                ),

                "decoder_dims": [
                    trial.suggest_int("dec1", 16, 64)
                ],

                "beta": trial.suggest_float(
                    "beta",
                    0.1,
                    2.0
                ),
            },

            "training": {

                "optimizer": sample_optimizer(
                    trial,
                    tun_cfg.training_space.optimizer
                ),

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

                "type": "vae"

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

    def build_preprocessor(self, prep_cfg):

        steps = []

        # scaler
        scaler_cfg = prep_cfg["scaler"]

        scaler = create_scaler(
            scaler_cfg["name"],
            **scaler_cfg.get("params", {}),
        )

        steps.append(scaler)

        return PreprocessingPipeline(steps)



    def adapt_input(self, X):
        # should be (N, T*F)
        return X.reshape(X.shape[0], -1)



    def build(
        self,
        cfg_model,
        cfg_training,
        input_shape,
        #training_context,
    ):
        model = self._build_model(
            cfg_model,
            input_shape,
        )

        trainer = self._build_trainer(
            cfg_training,
            model,
            #training_context,
        )

        return VAEWrapper(
            model,
            trainer,
        )


    """
    def build_training_context(
        self,
        cfg_training,
        y_train,
    ):
        return TrainingContext()
    """


    def _build_model(
        self,
        cfg_model,
        input_shape,
    ):
        input_dim = input_shape[1]

        model_cfg = VAEConfig(
            input_dim=input_dim,
            encoder_dims=cfg_model["encoder_dims"],
            latent_dim=cfg_model["latent_dim"],
            decoder_dims=cfg_model["decoder_dims"],
            beta=cfg_model["beta"],
        )

        return VAE(model_cfg)


    def _build_trainer(
        self,
        cfg_training,
        model,
        #training_context,
    ):
        optimizer = self._build_optimizer(
            cfg_training["optimizer"],
            model,
        )

        callbacks = create_callbacks(
            cfg_training["callbacks"]
        )

        trainer_cfg = TrainingConfig(
            epochs=cfg_training["epochs"],
            batch_size=cfg_training["batch_size"],
            optimizer=optimizer,
            loss=None,
            callbacks=callbacks,
        )

        trainer_cls = TRAINER_REGISTRY[
            cfg_training["type"]
        ]

        return trainer_cls(trainer_cfg)


    def _build_optimizer(
        self,
        cfg_optimizer,
        model,
    ):
        return create_optimizer(
            cfg_optimizer,
            model.parameters(),
        )




    def load(self, path):
        return VAEWrapper.load(path)