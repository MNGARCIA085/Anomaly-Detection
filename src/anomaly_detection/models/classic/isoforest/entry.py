

from sklearn.ensemble import IsolationForest
from anomaly_detection.models.registry import register
from anomaly_detection.preprocessing.pipeline import PreprocessingPipeline
from .model import IsoWrapper
from ...base_entry import BaseModelEntry
from anomaly_detection.preprocessing.components.scalers import create_scaler
from anomaly_detection.preprocessing.components.dimensionality import create_dimensionality_reducer
from anomaly_detection.preprocessing.components.feature_selection import create_feature_selector



from anomaly_detection.tuning.sample_prep import (
        sample_window_size, sample_feature_selection, sample_scaler, 
        sample_dimensionality_reducer
    )




@register("isoforest")
class IsoEntry(BaseModelEntry):


    def sample(self, trial, tun_cfg):
        # right now im not sampling threholders, but nmaybe later!!!!!!


        return {

            "name": "isoforest",

            "data": {
                "windowing": {
                    "size": sample_window_size(
                        trial,
                        tun_cfg.data.windowing.size,
                    ),
                },
            },

            "prep": {
                "feature_selection": sample_feature_selection(
                    trial,
                    tun_cfg.prep.feature_selection,
                ),
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
            },

            
        }


    def build_preprocessor(self, prep_cfg):

        steps = []

        # Feature selector; it goes first because im using Variance Thresold
        # in other situations it might go after scaling
        fs_cfg = prep_cfg["feature_selection"]

        if fs_cfg["enabled"]:
            selector = create_feature_selector(
                fs_cfg["name"],
                **fs_cfg.get("params", {}),
            )
            steps.append(selector)


        # scaler
        scaler_cfg = prep_cfg["scaler"]

        scaler = create_scaler(
            scaler_cfg["name"],
            **scaler_cfg.get("params", {}),
        )


        steps.append(scaler)


        # Dimensionality reducer
        reducer = create_dimensionality_reducer(
            prep_cfg["dimensionality"]
        )

        if reducer is not None:
            steps.append(reducer)


        return PreprocessingPipeline(
            steps
        )


    def adapt_input(self, X):
        # should be (N, T*F)
        return X.reshape(X.shape[0], -1)




    def build(
        self,
        model_cfg,
        training_cfg=None,
        input_shape=None,
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
