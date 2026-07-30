


#-------logger.py
from abc import ABC, abstractmethod


class ExperimentLogger(ABC):

    @abstractmethod
    def start_run(self, run_name=None):
        pass

    @abstractmethod
    def end_run(self):
        pass

    @abstractmethod
    def log_params(self, params):
        pass

    @abstractmethod
    def log_metrics(self, metrics):
        pass

    @abstractmethod
    def log_artifact(self, path):
        pass

    @abstractmethod
    def log_model(self, model, path):
        pass



#-------------mlflow logger
import os
from pathlib import Path
import mlflow
import joblib

from .logger import ExperimentLogger



import numpy as np 
import numbers


import joblib
import tempfile



from pathlib import Path




import pandas as pd
import matplotlib.pyplot as plt



class MLFlowLogger(ExperimentLogger):

    def __init__(
        self,
        exp_name='an_detection',
        tracking_db="mlflow.db",
        artifact_dir="mlruns"
    ):

        self.root_dir = Path(__file__).resolve().parents[3]

        self.tracking_db = (
            self.root_dir / tracking_db
        )

        self.artifact_dir = (
            self.root_dir / artifact_dir
        )

        self.exp_name = exp_name

        self._init_mlflow()


    def _init_mlflow(self):

        mlflow.set_tracking_uri(
            f"sqlite:///{self.tracking_db}"
        )

        os.makedirs(
            self.artifact_dir,
            exist_ok=True
        )

        mlflow.set_experiment(
            self.exp_name
        )


    def start_run(
        self,
        run_name=None
    ):

        return mlflow.start_run(
            run_name=run_name
        )


    def end_run(self):

        mlflow.end_run()


    def log_params(
        self,
        params
    ):

        mlflow.log_params(
            params
        )




    """
    def log_metrics(
        self,
        metrics
    ):

        mlflow.log_metrics(
            metrics
        )
    """


    def log_metrics(self, metrics):

        clean_metrics = {}

        for k, v in metrics.items():

            if isinstance(v, np.generic):
                v = v.item()

            if isinstance(v, numbers.Number):
                clean_metrics[k] = float(v)

        mlflow.log_metrics(clean_metrics)



    def log_artifactv0(
        self,
        path
    ):

        mlflow.log_artifact(
            str(path)
        )



    def log_artifact(
        self,
        path,
        artifact_path=None
    ):

        path = Path(path)

        if path.is_dir():
            mlflow.log_artifacts(
                str(path),
                artifact_path=artifact_path
            )

        else:
            mlflow.log_artifact(
                str(path),
                artifact_path=artifact_path
            )



    def artifact_path(self, filename):

        run_id = mlflow.active_run().info.run_id

        path = (
            self.root_dir
            / "artifacts"
            / run_id
        )

        path.mkdir(
            parents=True,
            exist_ok=True
        )

        return path / filename


    def log_preprocessor(
        self,
        preprocessor
    ):

        path = self.artifact_path(
            "preprocessor.pkl"
        )

        preprocessor.save(path)

        mlflow.log_artifact(
            str(path),
            artifact_path="preprocessing"
        )




    def log_model(
        self,
        model,
        path
    ):

        model_path = (
            self.root_dir / path
        )

        joblib.dump(
            model,
            model_path
        )

        mlflow.log_artifact(
            str(model_path)
        )






    def log_training_history(
        self,
        history
    ):

        if not history.metrics:
            return

        # ---------- CSV ----------
        df = pd.DataFrame(history.as_dict())

        csv_path = self.artifact_path(
            "history.csv"
        )

        df.to_csv(
            csv_path,
            index=False
        )

        self.log_artifact(
            csv_path,
            artifact_path="training"
        )


        # ---------- One plot per metric ----------
        for metric, values in history.as_dict().items():

            plt.figure()

            plt.plot(
                range(1, len(values) + 1),
                values
            )

            plt.xlabel("Epoch")
            plt.ylabel(metric)
            plt.title(metric)

            fig_path = self.artifact_path(
                f"{metric}.png"
            )

            plt.savefig(
                fig_path,
                bbox_inches="tight"
            )

            plt.close()

            self.log_artifact(
                fig_path,
                artifact_path="training"
            )


        # ---------- Combined plot ----------
        plt.figure()

        for metric, values in history.as_dict().items():

            plt.plot(
                range(1, len(values) + 1),
                values,
                label=metric
            )

        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Training History")
        plt.legend()

        fig_path = self.artifact_path(
            "history.png"
        )

        plt.savefig(
            fig_path,
            bbox_inches="tight"
        )

        plt.close()

        self.log_artifact(
            fig_path,
            artifact_path="training"
        )





# helper
def flatten_dict(d, parent_key=""):
    items = {}

    for k, v in d.items():
        key = f"{parent_key}.{k}" if parent_key else k

        if isinstance(v, dict):
            items.update(
                flatten_dict(v, key)
            )

        elif isinstance(v, list):
            items[key] = str(v)

        else:
            items[key] = v

    return items






"""
import mlflow



class ExperimentLogger:
    def __init__(self, metric_name="val_score", mode="max"):
        self.metric_name = metric_name
        self.mode = mode
        self.best = None

    def _is_better(self, value):
        if self.best is None:
            return True
        if self.mode == "max":
            return value > self.best
        return value < self.best

    def log_metrics(self, metrics: dict, step=None):
        mlflow.log_metrics(metrics, step=step)

    def maybe_log_model(self, model, pipeline, metrics: dict):
        value = metrics[self.metric_name]

        if self._is_better(value):
            self.best = value

            print(f"New best {self.metric_name}: {value:.4f} → saving")

            # save locally
            import pickle
            with open("pipeline.pkl", "wb") as f:
                pickle.dump(pipeline, f)

            with open("model.pkl", "wb") as f:
                pickle.dump(model, f)

            # log to MLflow
            mlflow.log_artifact("pipeline.pkl")
            mlflow.log_artifact("model.pkl")

            mlflow.set_tag("best_model", "true")

"""




"""

with mlflow.start_run():

    logger = ExperimentLogger(metric_name="val_auc", mode="max")

    for epoch in range(n_epochs):

        # train...
        metrics = {
            "train_loss": ...,
            "val_auc": ...
        }

        logger.log_metrics(metrics, step=epoch)

        logger.maybe_log_model(model, prep, metrics)



with my own fn:

logger = ExperimentLogger(
    metric_name="val_f1",
    is_better=lambda new, best: new > best
)


# get best
from mlflow.tracking import MlflowClient

def get_best_from_mlflow(experiment_name, metric_name, mode="max"):
    client = MlflowClient()

    exp = client.get_experiment_by_name(experiment_name)

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=[f"metrics.{metric_name} DESC"] if mode=="max"
                 else [f"metrics.{metric_name} ASC"],
        max_results=1
    )

    if not runs:
        return None

    return runs[0].data.metrics.get(metric_name)



class ExperimentLogger:

    def __init__(self, experiment_name, metric_name, is_better):
        self.metric_name = metric_name
        self.is_better = is_better
        self.experiment_name = experiment_name

        self.best_global = get_best_from_mlflow(
            experiment_name,
            metric_name
        )

    def maybe_log_model(self, model, pipeline, metrics):
        current = metrics[self.metric_name]

        if self.best_global is None or self.is_better(current, self.best_global):

            print(f"New GLOBAL best: {current:.4f}")

            self.best_global = current

            # save + log artifacts
            ...

            mlflow.set_tag("best_model", "true")


"""


"""
load best

from mlflow.tracking import MlflowClient

def get_best_run(experiment_name, metric_name, mode="max"):
    client = MlflowClient()

    exp = client.get_experiment_by_name(experiment_name)

    order = "DESC" if mode == "max" else "ASC"

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=[f"metrics.{metric_name} {order}"],
        max_results=1
    )

    if not runs:
        raise ValueError("No runs found")

    return runs[0]


load artifacts from that run

import mlflow
import pickle

def load_artifact_from_run(run_id, artifact_path):
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path=artifact_path
    )

    with open(local_path, "rb") as f:
        return pickle.load(f)



fn to rule them all

def load_best_pipeline(experiment_name, metric_name, mode="max"):
    best_run = get_best_run(experiment_name, metric_name, mode)

    run_id = best_run.info.run_id

    print(f"Loading best run: {run_id}")

    pipeline = load_artifact_from_run(run_id, "pipeline.pkl")

    return pipeline

pipeline = load_best_pipeline(
    experiment_name="my-exp",
    metric_name="val_f1",
    mode="max"
)

X_new = pipeline.transform(X_new)



if you also save the model

def load_best_predictor(experiment_name, metric_name, mode="max"):
    best_run = get_best_run(experiment_name, metric_name, mode)
    run_id = best_run.info.run_id

    pipeline = load_artifact_from_run(run_id, "pipeline.pkl")
    model = load_artifact_from_run(run_id, "model.pkl")

    return pipeline, model

pipeline, model = load_best_predictor("my-exp", "val_f1")

X = pipeline.transform(X)
y_pred = model.predict(X)

mlflow.log_metric("val_f1", value)

"""


"""
You now have:

training → logs everything
logger → decides what to save
MLflow → ranks runs
loader → retrieves best

👉 That’s a complete experiment loop
"""


"""
for diff. types

def maybe_log_model(self, model, pipeline, metrics, extra_log_fn=None):

    current = metrics[self.metric_name]

    if self.best_global is None or self.is_better(current, self.best_global):

        self.best_global = current

        # core artifacts
        save_pickle(pipeline, "pipeline.pkl")
        save_pickle(model, "model.pkl")

        mlflow.log_artifact("pipeline.pkl")
        mlflow.log_artifact("model.pkl")

        # 🔥 extra artifacts (model-specific)
        if extra_log_fn:
            extra_log_fn()

def log_nn_artifacts():
    mlflow.log_artifact("loss_curve.png")
    mlflow.log_artifact("metrics.json")

logger.maybe_log_model(
    model,
    prep,
    metrics,
    extra_log_fn=log_nn_artifacts
)

"""


"""
Analyze later
#--------MLFLOW logging-----------#
class MLflowLogger:

    def log_artifacts(self, artifacts):
        for name, obj in artifacts.items():
            joblib.dump(obj, f"{name}.pkl")
            mlflow.log_artifact(f"{name}.pkl")
"""