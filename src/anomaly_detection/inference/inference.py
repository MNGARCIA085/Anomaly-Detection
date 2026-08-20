import numpy as np

from pathlib import Path
import joblib

import anomaly_detection.models.register_models 
from anomaly_detection.models.registry import MODEL_REGISTRY

from anomaly_detection.data.windowing import Windowing

from anomaly_detection.thresholding.thresholding import (
        Thresholding
    )

from mlflow.tracking import MlflowClient




class InferenceRunner:

    def __init__(
        self,
        prep,
        windowing,
        entry,
        wrapper,
        thresholding=None,
    ):
        self.prep = prep
        self.windowing = windowing
        self.entry = entry
        self.wrapper = wrapper
        self.thresholding = thresholding

    def predict(self, X):

        X_p = self.prep.transform(X)

        X_w = self.windowing.transform(X_p)

        X_model = self.entry.adapt_input(X_w)

        if self.thresholding is not None:

            threshold = (
                self.thresholding.get_threshold()
            )

            return self.wrapper.predict(
                X_model,
                threshold,
            )

        return self.wrapper.predict(X_model)








def load_inference_runner(run_id):

    client = MlflowClient()

    run = client.get_run(run_id)

    local_dir = Path(
        client.download_artifacts(
            run_id,
            "",
        )
    )

    # Preprocessor
    prep = joblib.load(
        local_dir
        / "preprocessing"
        / "preprocessor.pkl"
    )

    # windowing
    windowing = Windowing(int(run.data.params["data.windowing.size"]))

    # Model
    model_type = run.data.tags["model_type"]

    entry = MODEL_REGISTRY[model_type]()

    wrapper = entry.load(
        local_dir / "model"
    )

    # Threshold
    thresholding = None

    thresholding_path = (
        local_dir
        / "thresholding"
        / "thresholding.pkl"
    )

    if thresholding_path.exists():

        thresholding = Thresholding.load(
            thresholding_path
        )



    return InferenceRunner(
        prep=prep,
        windowing=windowing,
        entry=entry,
        wrapper=wrapper,
        thresholding=thresholding,
    )








"""
For the real benchmark, replace the generated X_benchmark with a fixed subset of your actual 
raw validation data:

X_benchmark = X_val[:100]

That is preferable to random data because every candidate sees the same realistic input distribution.

And raw_input_dim should come from the raw dataset, not from any candidate's preprocessor.
"""