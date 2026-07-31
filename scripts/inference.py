from pathlib import Path

import joblib
import mlflow
import numpy as np
from mlflow.tracking import MlflowClient

from anomaly_detection.models.nnets.ae.model import AEWrapper










#----------log model_type!!!!!!!!!!!!


def main(run_id):

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    client = MlflowClient()

    # -----------------------
    # Run information
    # -----------------------

    run = client.get_run(run_id)

    print("\n=== PARAMETERS ===")
    for k, v in run.data.params.items():
        print(f"{k}: {v}")

    print("\n=== METRICS ===")
    for k, v in run.data.metrics.items():
        print(f"{k}: {v}")

    # -----------------------
    # Download artifacts
    # -----------------------

    local_dir = Path(
        client.download_artifacts(
            run_id,
            ""
        )
    )

    print("\nArtifacts downloaded to:")
    print(local_dir)

    # -----------------------
    # Load preprocessor
    # -----------------------

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
            print("PCA components =", step.components_.shape)

    # -----------------------
    # Load model
    # -----------------------

    """
    model_type = run.data.params["model_type"]

    entry = MODEL_REGISTRY[model_type]()

    wrapper = entry.load(local_dir / "model")
    """


    # add later model type to logged params in mlflow

    # for the registry to work, improve later
    from anomaly_detection.models.nnets.ae.entry import AEEntry
    from anomaly_detection.models.classic.isoforest.entry import IsoEntry
    
    # registry
    from anomaly_detection.models.registry import MODEL_REGISTRY


    entry = MODEL_REGISTRY['ae']() # hardcdoed for now; "ae"

    wrapper = entry.load(local_dir / "model")


    #wrapper = AEWrapper.load(
    #    local_dir / "model"
    #)

    print("\n=== MODEL ===")
    print(wrapper.model)

    # -----------------------
    # Quick inference test
    # -----------------------

    """ AEs
    X = np.random.randn(
        5,
        wrapper.model.config.input_dim # only for aes
    )

    X = prep.transform(X)

    scores = wrapper.get_scores(X)
    """




    #
    X = np.random.randn(
        5,
        wrapper.input_dim # only for aes
    )

    X = prep.transform(X)

    scores = wrapper.get_scores(X)





    print("\n=== SCORES ===")
    print(scores)




if __name__=="__main__":
    main("6ca321accc624c679bf612bd1af43506")
    #main("8c10dc66d8cb4153a4d2b9b738923acc") # iso