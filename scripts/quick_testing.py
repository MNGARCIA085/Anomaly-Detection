from anomaly_detection.data.data import DataModule
    
from anomaly_detection.infra.selection.candidate_registry import CandidateRegistry
from anomaly_detection.infra.selection.model_selector import ModelSelector
from pathlib import Path

from anomaly_detection.inference.benchmarking import benchmark_candidates


BASE_DIR = Path(__file__).resolve().parents[1]  # __file__ -> actual file location
TRAIN_PATH = BASE_DIR / "data" / "servers" / "X_part2.npy"
VAL_PATH = BASE_DIR / "data" / "servers" / "X_val_part2.npy"
Y_VAL_PATH = BASE_DIR / "data" / "servers" / "y_val_part2.npy"




def main():



    # load data

    data = DataModule(TRAIN_PATH, VAL_PATH, Y_VAL_PATH)
    X_train, X_val, y_val = data.load()



    root_dir = Path(__file__).resolve().parents[1]
    tracking_db = root_dir / "mlflow.db"
    candidate_db_url = f"sqlite:///{tracking_db}"

   
    registry = CandidateRegistry(candidate_db_url)

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
        X_benchmark=X_val[:100], # later the 100 came from config
    )


    selected = selector.select(
        experiment_id=1,
        pr_auc_tolerance=0.005, # later from config!!!
    )


    print(selected)
    print(selected.run_id)
    print(selected.model_family)
    print(selected.val_pr_auc)





if __name__ == "__main__":
    main()