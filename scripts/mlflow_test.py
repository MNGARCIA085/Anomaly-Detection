from pathlib import Path
import joblib
import mlflow
from mlflow.tracking import MlflowClient






def main(run_id):
    # -----------------------
    # Configuration
    # -----------------------

    RUN_ID = run_id

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    client = MlflowClient()


    # -----------------------
    # Run information
    # -----------------------

    run = client.get_run(RUN_ID)

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
            RUN_ID,
            ""
        )
    )

    print("\nArtifacts downloaded to:")
    print(local_dir)


    # -----------------------
    # Load preprocessor
    # -----------------------


    # if i set art. path properly
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

    print("\n=== MODEL ===")

    model_dir = local_dir / "model"

    print(model_dir)




if __name__=="__main__":
    main("46766beb5da344e5a29b813e2f7884c6")