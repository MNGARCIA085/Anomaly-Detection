from anomaly_detection.models.registry import register
from .schemas import TransformerAEConfig
from anomaly_detection.training.schemas import TrainingConfig
from anomaly_detection.preprocessing.pipeline import PreprocessingPipeline
from .model import TransformerAE, TransformerAEWrapper
from anomaly_detection.training.registry import TRAINER_REGISTRY
from anomaly_detection.models.base_entry import BaseModelEntry
from anomaly_detection.preprocessing.components.scalers import create_scaler




from anomaly_detection.training.callbacks.registry import create_callbacks
from anomaly_detection.training.optimizers.registry import create_optimizer
from anomaly_detection.tuning.sample_training import sample_callbacks, sample_optimizer


from anomaly_detection.tuning.sample_prep import sample_window_size,sample_scaler



from anomaly_detection.training.losses import create_loss


"""
There is one thing to adjust compared with your VAE entry: 
self.seq_len needs to come from your actual pipeline/config. 
I would not invent it inside the entry.
model_cfg = TransformerAEConfig(
    input_dim=input_dim,
    seq_len=seq_len,
    ...
)
where seq_len is the window length established by your data/windowing configuration.
"""





@register("transformer")
class TransformerEntry(BaseModelEntry):

    def sample(self, trial, tun_cfg):

        return {

            "name": "transformer",



            "data": {
                "windowing": {
                    "size": sample_window_size(
                        trial,
                        tun_cfg.data.windowing.size,
                    ),
                },
            },

            # ========================================================
            # Preprocessing
            # ========================================================

            "prep": {
                "scaler": sample_scaler(
                    trial,
                    tun_cfg.prep.scaler,
                ),
            },

            # ========================================================
            # Model
            # ========================================================

            "models": {

                "d_model": trial.suggest_categorical(
                    "d_model",
                    [32, 64, 128],
                ),

                "nhead": trial.suggest_categorical(
                    "nhead",
                    [2, 4, 8],
                ),

                "num_encoder_layers": trial.suggest_int(
                    "num_encoder_layers",
                    1,
                    4,
                ),

                "dim_feedforward": trial.suggest_categorical(
                    "dim_feedforward",
                    [64, 128, 256],
                ),

                "dropout": trial.suggest_float(
                    "dropout",
                    0.0,
                    0.3,
                ),
            },

            # ========================================================
            # Training
            # ========================================================

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

                "type": "default",
            },

            # ========================================================
            # Thresholding
            # ========================================================

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

    # ================================================================
    # Preprocessing
    # ================================================================

    def build_preprocessor(self, prep_cfg):

        steps = []

        scaler_cfg = prep_cfg["scaler"]

        scaler = create_scaler(
            scaler_cfg["name"],
            **scaler_cfg.get("params", {}),
        )

        steps.append(scaler)

        return PreprocessingPipeline(steps)

    # ================================================================
    # Input adaptation
    # ================================================================

    def adapt_input(self, X):

        # Transformer expects:
        # (N, T, F)
        #
        # N = number of windows
        # T = sequence length
        # F = number of features

        return X

    # ================================================================
    # Model / Trainer
    # ================================================================

    def build(
        self,
        cfg_model,
        cfg_training,
        input_shape,
    ):

        input_dim = input_shape[-1]
        seq_len = input_shape[1]

        model_cfg = TransformerAEConfig(
            input_dim=input_dim,
            seq_len=seq_len,
            d_model=cfg_model["d_model"],
            nhead=cfg_model["nhead"],
            num_encoder_layers=cfg_model["num_encoder_layers"],
            dim_feedforward=cfg_model["dim_feedforward"],
            dropout=cfg_model["dropout"],
        )

        model = TransformerAE(model_cfg)

        optimizer = create_optimizer(
            cfg_training["optimizer"],
            model.parameters()
        )

        # loss
        loss = create_loss(
            cfg_training["loss"],
        )

        # callbacks
        callbacks = create_callbacks(cfg_training["callbacks"])

        trainer_cfg = TrainingConfig(
            epochs=cfg_training["epochs"],
            batch_size=cfg_training["batch_size"],
            optimizer=optimizer,
            loss=loss,
            callbacks=callbacks,
        )

        trainer_cls = TRAINER_REGISTRY[
            cfg_training["type"]
        ]

        trainer = trainer_cls(trainer_cfg)

        return TransformerAEWrapper(
            model,
            trainer
        )

    # ================================================================
    # Load
    # ================================================================

    def load(self, path):

        return TransformerAEWrapper.load(path)