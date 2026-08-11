from pathlib import Path

import joblib
import mlflow
import numpy as np
from mlflow.tracking import MlflowClient

from anomaly_detection.models.nnets.ae.model import AEWrapper


import anomaly_detection.models.register_models 
from anomaly_detection.models.registry import MODEL_REGISTRY




#----------log model_type-------------------#
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
    model_type = run.data.tags['model_type']

    entry = MODEL_REGISTRY[model_type]() # hardcdoed for now; "ae"

    wrapper = entry.load(local_dir / "model")


    print("\n=== MODEL ===")
    print(wrapper.model)

    # -----------------------
    # Quick inference test
    # -----------------------


    print('input dim', prep.input_dim)



    # fake data
    X = np.random.randn(
        5,
        #wrapper.input_dim # final input: ex:8
        #11
        prep.input_dim
    )

    X = prep.transform(X)

    scores = wrapper.get_scores(X)

    print("\n=== SCORES ===")
    print(scores)




if __name__=="__main__":
    main("c3fdbab89bdb4740b78d0f76d0a92623") # ae
    main("217a20b1284d452fa2adf793a364b4e2") # iso
    main("74fe47e70b834f468c762d0296b47361") # iso