from pathlib import Path

import joblib
import mlflow
import numpy as np
from mlflow.tracking import MlflowClient

from anomaly_detection.models.nnets.ae.model import AEWrapper


import anomaly_detection.models.register_models 
from anomaly_detection.models.registry import MODEL_REGISTRY



from anomaly_detection.data.windowing import Windowing


#----------log model_type-------------------#
def main(run_id):

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    client = MlflowClient()

    # ============================================================
    # Run information
    # ============================================================

    run = client.get_run(run_id)

    print("\n=== PARAMETERS ===")

    for k, v in run.data.params.items():
        print(f"{k}: {v}")

    print("\n=== METRICS ===")

    for k, v in run.data.metrics.items():
        print(f"{k}: {v}")

    # ============================================================
    # Download artifacts
    # ============================================================

    local_dir = Path(
        client.download_artifacts(
            run_id,
            ""
        )
    )

    print("\nArtifacts downloaded to:")
    print(local_dir)

    # ============================================================
    # Load preprocessor
    # ============================================================

    prep = joblib.load(
        local_dir / "preprocessing" / "preprocessor.pkl"
    )

    print("\n=== PREPROCESSOR ===")

    for step in prep.steps:

        print(type(step).__name__)

        if hasattr(step, "mean_"):
            print("mean =", step.mean_)

        if hasattr(step, "scale_"):
            print("scale =", step.scale_)

        if hasattr(step, "components_"):
            print(
                "PCA components =",
                step.components_.shape
            )

    # ============================================================
    # Load model
    # ============================================================

    model_type = run.data.tags["model_type"]

    entry = MODEL_REGISTRY[model_type]()

    wrapper = entry.load(
        local_dir / "model"
    )

    print("\n=== MODEL TYPE ===")
    print(model_type)

    print("\n=== MODEL ===")
    print(wrapper.model)

    # ============================================================
    # Load thresholding
    # ============================================================

    from anomaly_detection.thresholding.thresholding import (
        Thresholding
    )

    thresholding_path = (
        local_dir
        / "thresholding"
        / "thresholding.pkl"
    )

    thresholding = None

    if thresholding_path.exists():

        thresholding = Thresholding.load(
            thresholding_path
        )

        print("\n=== THRESHOLD ===")

        print(
            thresholding.get_threshold()
        )

    # ============================================================
    # Inference data
    # ============================================================

    print("\n=== INFERENCE ===")

    print(
        "Preprocessor input dim:",
        prep.input_dim
    )

    # Fake raw data
    #
    # Number of features must match the data expected
    # by the saved preprocessor.

    X = np.random.randn(
        20,
        prep.input_dim
    )

    print(
        "Raw X:",
        X.shape
    )

    # ============================================================
    # Preprocessing
    # ============================================================

    X_p = prep.transform(X)

    print(
        "Preprocessed X:",
        X_p.shape
    )

    # ============================================================
    # Windowing
    # ============================================================

    # Temporary hardcoded window size.
    # Later this should come from configuration/model metadata.

    windowing = Windowing(10)

    X_w = windowing.transform(
        X_p
    )

    print(
        "Windowed X:",
        X_w.shape
    )

    # ============================================================
    # Model-specific representation
    # ============================================================

    # AE / Isolation Forest:
    #
    #     (N, T, F)
    #          ↓
    #     (N, T * F)
    #
    # Transformer:
    #
    #     (N, T, F)
    #          ↓
    #     (N, T, F)

    X_model = entry.adapt_input(
        X_w
    )

    print(
        "Model input X:",
        X_model.shape
    )

    # ============================================================
    # Scores
    # ============================================================

    scores = wrapper.get_scores(
        X_model
    )

    print("\n=== SCORES ===")
    print(scores)

    # ============================================================
    # Predictions
    # ============================================================

    if thresholding is not None:

        threshold = (
            thresholding.get_threshold()
        )

        predictions = wrapper.predict(
            X_model,
            threshold,
        )

    else:

        # Models with native prediction mechanisms,
        # such as Isolation Forest.

        predictions = wrapper.predict(
            X_model
        )

    print("\n=== PREDICTIONS ===")
    print(predictions)



if __name__=="__main__":
    main("46dc952236e74b5480534f93c21d637c") # ae
    main("e9cbacbd3be84ed6bbea10f3c0e8c7c4") # transf.
    #main("42dfd892996a411d9463f7640367c468") # iso
    #main("0153f371b3a04505a984b701b9b78060") # vae




"""
0 = normal
1 = anomaly
"""


