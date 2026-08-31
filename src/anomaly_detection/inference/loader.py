import numpy as np
from pathlib import Path
import joblib

import anomaly_detection.models.register_models 
from anomaly_detection.models.registry import MODEL_REGISTRY

from anomaly_detection.data.windowing import Windowing

from anomaly_detection.thresholding.thresholding import (
        Thresholding
    )


from .runner import InferenceRunner





def _build_runner(
    model_dir,
    model_type,
    window_size,
):

    prep = joblib.load(
        model_dir
        / "preprocessing"
        / "preprocessor.pkl"
    )

    windowing = Windowing(window_size)


    # temporal prep
    temporal_prep = None

    temporal_prep_path = (
        model_dir
        / "temporal_preprocessing"
        / "temporal_preprocessor.pkl"
    )

    if temporal_prep_path.exists():

        temporal_prep = joblib.load(
            temporal_prep_path,
        )



    entry = MODEL_REGISTRY[model_type]()



    wrapper = entry.load(
        model_dir / "model"
    )



    # thresholding
    thresholding = None

    thresholding_path = (
        model_dir
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
        temporal_prep=temporal_prep,
        thresholding=thresholding,
    )





def load_from_config(config, model_dir):
    return _build_runner(
        model_dir=Path(model_dir),
        model_type=config["model"]["type"],
        window_size=config["data"]["windowing"]["size"],
    )

