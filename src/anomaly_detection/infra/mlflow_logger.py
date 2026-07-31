import os
from pathlib import Path
import mlflow
import joblib

from .logger import ExperimentLogger

import numpy as np 
import numbers

import pandas as pd
import matplotlib.pyplot as plt

from typing import Optional


class MLFlowLogger(ExperimentLogger):

    def __init__(
        self,
        exp_name='Anomaly_Detection',
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



    def log_tags(
        self,
        model_type: str,
        dataset: Optional[str] = None,
        trainer: Optional[str] = None,
        framework: Optional[str] = None,
        **extra_tags,
    ) -> None:
        """Log metadata tags for the current MLflow run."""

        tags = {
            "model_type": model_type,
            "dataset": dataset,
            "trainer": trainer,
            "framework": framework,
            **extra_tags,
        }

        # Remove unset values
        tags = {k: v for k, v in tags.items() if v is not None}

        mlflow.set_tags(tags)


    def log_params(
        self,
        params
    ):
        mlflow.log_params(
            params
        )


    def log_metrics(self, metrics):

        clean_metrics = {}

        for k, v in metrics.items():

            if isinstance(v, np.generic):
                v = v.item()

            if isinstance(v, numbers.Number):
                clean_metrics[k] = float(v)

        mlflow.log_metrics(clean_metrics)



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


    # training history
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


"""
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000


rm -rf mlruns/
rm -rf mlartifacts/      # if it exists
rm -f mlflow.db          # if using SQLite

if I save outside
rm -rf artifacts/
rm -rf checkpoints/
rm -rf saved_models/


"""