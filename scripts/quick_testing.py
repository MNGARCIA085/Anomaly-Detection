from anomaly_detection.infra.candidate_models import CandidateRegistry, ModelSelector
from pathlib import Path



from anomaly_detection.inference.inference import benchmark_candidates


def main():

    root_dir = Path(__file__).resolve().parents[1]
    tracking_db = root_dir / "mlflow.db"

   
    registry = CandidateRegistry(tracking_db)

    registry.print_candidates(
        experiment_id=1,
        include_evicted=True,
    )


    print("selected")

    selector = ModelSelector(
        registry
    )


    #------------------
    benchmark_candidates(
        registry=registry,
        experiment_id=1,
        #raw_input_dim=X_train.shape[1],
        raw_input_dim=11
    )


    selected = selector.select(
        experiment_id=1,
        pr_auc_tolerance=0.005, # later from config!!!
    )


    print(selected)
    print(selected["run_id"])
    print(selected["model_family"])
    print(selected["val_pr_auc"])





if __name__ == "__main__":
    main()