from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from anomaly_detection.models.registry import register



from anomaly_detection.preprocessing.pipeline import PreprocessingPipeline



from .model import IsoWrapper


from ...base_entry import BaseModelEntry




@register("isoforest")
class IsoEntry(BaseModelEntry):


    def sample(self, trial, tun_cfg):


        return {

            "prep": {
                "scaler": trial.suggest_categorical(
                    "scaler",
                    ["standard", "minmax"]
                )
            },

            "models": {

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


    def build_preprocessor(self, prep_cfg):

        steps = []

        if prep_cfg["scaler"] == "standard":
            steps.append(StandardScaler())

        elif prep_cfg["scaler"] == "minmax":
            steps.append(MinMaxScaler())

        return PreprocessingPipeline(
            steps
        )


    def build(
        self,
        model_cfg,
        training_cfg=None,
        input_dim=None,
    ):


        # my model, not need to define it custom like AE
        model = IsolationForest(
            n_estimators=model_cfg["n_estimators"],
            contamination=model_cfg["contamination"]
        )

        return IsoWrapper(
            model
        )


    # to make inference easier
    # The entry is just delegating. It's a tiny method, but it keeps the pipeline generic.
    def load(self, path):

        return IsoWrapper.load(path)
