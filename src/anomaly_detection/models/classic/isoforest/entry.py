from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from anomaly_detection.models.registry import register



from anomaly_detection.preprocessing.pipeline import PreprocessingPipeline



from .model import IsoWrapper




@register("isoforest") # iso
class IsoEntry:

    @staticmethod
    def sample(trial, tun_cfg):

        print(tun_cfg)

        return {

            "prep": {
                "scaler": trial.suggest_categorical(
                    "scaler",
                    ["standard", "minmax"]
                )
            },

            "model": {

                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    tun_cfg.model_space.n_estimators.low,
                    tun_cfg.model_space.n_estimators.high
                ),

                "contamination": trial.suggest_float(
                    "contamination",
                    tun_cfg.model_space.contamination.low,
                    tun_cfg.model_space.contamination.high
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

        return PreprocessingPipeline(
            steps
        )

    @staticmethod
    def build(
        cfg,
        input_dim
    ):

        # my model, not need to define it custom like AE
        model = IsolationForest(
            n_estimators=cfg["model"]["n_estimators"],
            contamination=cfg["model"]["contamination"]
        )

        return IsoWrapper(
            model
        )