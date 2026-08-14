
#https://chatgpt.com/c/6a74e018-c794-83e9-a42a-99b566358b62


from anomaly_detection.models.registry import register
from .schemas import VAEConfig
from anomaly_detection.training.schemas import TrainingConfig



from anomaly_detection.preprocessing.pipeline import PreprocessingPipeline



from .model import VAE, VAEWrapper


from anomaly_detection.training.callbacks import EarlyStopping,PrintLossCallback


from anomaly_detection.training.registry import TRAINER_REGISTRY


from ...base_entry import BaseModelEntry


# from anomaly_detection.training.losses import create_loss


from anomaly_detection.training.optimizers import sample_optimizer, create_optimizer








# no loss object, since the VAE computes reconstruction + KL internally.



from anomaly_detection.preprocessing.components.scalers import create_scaler, sample_scaler


@register("vae")
class VAEEntry(BaseModelEntry):

    def sample(self, trial, tun_cfg):

        return {

            "name": "vae",

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

    def build(
        self,
        cfg_model,
        cfg_training,
        input_dim
    ):


        model_cfg = VAEConfig(
            input_dim=input_dim,
            encoder_dims=cfg_model["encoder_dims"],
            latent_dim=cfg_model["latent_dim"],
            decoder_dims=cfg_model["decoder_dims"],
            beta=cfg_model["beta"],
        )

        model = VAE(model_cfg)

        optimizer = create_optimizer(
            cfg_training["optimizer"],
            model.parameters()
        )

        trainer_cfg = TrainingConfig(
            epochs=cfg_training["epochs"],
            batch_size=cfg_training["batch_size"],
            optimizer=optimizer,
            loss=None,
            callbacks=[
                EarlyStopping(patience=3),
                PrintLossCallback(),
            ]
        )

        trainer_cls = TRAINER_REGISTRY[cfg_training["type"]]
        trainer = trainer_cls(trainer_cfg)


        return VAEWrapper(
            model,
            trainer
        )

    def load(self, path):

        return VAEWrapper.load(path)