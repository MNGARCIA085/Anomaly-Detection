from sklearn.ensemble import IsolationForest
from anomaly_detection.models.registry import register
from anomaly_detection.preprocessing.pipeline import PreprocessingPipeline
from .model import IsoWrapper
from ...base_entry import BaseModelEntry
from anomaly_detection.preprocessing.components.scalers import create_scaler, sample_scaler
from anomaly_detection.preprocessing.components.dimensionality import create_dimensionality_reducer, sample_dimensionality_reducer



@register("isoforest")
class IsoEntry(BaseModelEntry):


    def sample(self, trial, tun_cfg):


        return {

            "name": "isoforest",

            "prep": {
                "scaler": sample_scaler(
                    trial,
                    tun_cfg.prep.scaler,
                ),
                "dimensionality": sample_dimensionality_reducer(
                    trial,
                    tun_cfg.prep.dimensionality,
                ),
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

        # scaler
        scaler_cfg = prep_cfg["scaler"]

        scaler = create_scaler(
            scaler_cfg["name"],
            **scaler_cfg.get("params", {}),
        )


        steps.append(scaler)

        print('\n', prep_cfg)


        # Dimensionality reducer
        reducer = create_dimensionality_reducer(
            prep_cfg["dimensionality"]
        )

        if reducer is not None:
            steps.append(reducer)


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
