import anomaly_detection.models.register_models # to trigger registration
from anomaly_detection.models.registry import MODEL_REGISTRY

from anomaly_detection.evaluation.evaluator import Evaluator

#from anomaly_detection.infra.utils import flatten_dict
#from anomaly_detection.infra.mlflow_logger import  MLFlowLogger


from anomaly_detection.infra.null_logger import NullLogger

from anomaly_detection.thresholding.thresholding import create_threshold_strategy
from anomaly_detection.thresholding.thresholding import Thresholding


from anomaly_detection.data.windowing import Windowing



# TRANSF.
# https://chatgpt.com/c/6a839478-0014-83e9-930c-04b35deb7350


class Experiment:

    def __init__(
        self,
        model_type,
        evaluator,
        logger=None,
    ):
        self.model_type = model_type
        self.evaluator = evaluator
        self.logger = logger or NullLogger()

    def run(
        self,
        cfg,
        X_train,
        X_val,
        y_val=None,
        run_type="train",  # train or tune
    ):

        with self.logger.start_run(
            run_name=self.model_type
        ):

            # --------------------------------------------------
            # 1. Build model entry
            # --------------------------------------------------

            entry = MODEL_REGISTRY[
                self.model_type
            ]()

            # --------------------------------------------------
            # 2. Preprocessing
            #    Fit ONLY on train
            # --------------------------------------------------

            preprocessor = (
                entry.build_preprocessor(
                    cfg.get("prep")
                )
            )

            X_train_p = (
                preprocessor.fit_transform(
                    X_train
                )
            )

            X_val_p = (
                preprocessor.transform(
                    X_val
                )
            )

            # --------------------------------------------------
            # 3. Windowing
            # --------------------------------------------------

            windowing = Windowing(
                cfg.get("windowing", {}).get("size", 10)
            )
            

            X_train_w = (
                windowing.transform(
                    X_train_p
                )
            )

            X_val_w, y_val_w = (
                windowing.transform_with_labels(
                    X_val_p,
                    y_val
                )
            )

            # --------------------------------------------------
            # 4. Model-specific representation
            #
            # AE:
            #     (samples, window, features)
            #         ->
            #     (samples, window * features)
            #
            # Transformer:
            #     probably keeps sequence representation
            #
            # IsoForest:
            #     flattened representation
            # --------------------------------------------------

            X_train_model = (
                entry.adapt_input(
                    X_train_w
                )
            )

            X_val_model = (
                entry.adapt_input(
                    X_val_w
                )
            )

            # --------------------------------------------------
            # 5. Input dimension AFTER adaptation
            # --------------------------------------------------

            #input_dim = (
            #    X_train_model.shape[1]
            #)
            #input_dim = X_train_model.shape[-1]
            input_shape = X_train_model.shape





            # --------------------------------------------------
            # Debug
            # --------------------------------------------------

            print(
                "Shapes:"
            )

            print(
                "  X_train_p:",
                X_train_p.shape
            )

            print(
                "  X_val_p:",
                X_val_p.shape
            )

            print(
                "  X_train_w:",
                X_train_w.shape
            )

            print(
                "  X_val_w:",
                X_val_w.shape
            )

            print(
                "  X_train_model:",
                X_train_model.shape
            )

            print(
                "  X_val_model:",
                X_val_model.shape
            )

            if y_val_w is not None:
                print(
                    "  y_val_w:",
                    y_val_w.shape
                )

            # --------------------------------------------------
            # 6. Build model
            # --------------------------------------------------

            wrapper = (
                entry.build(
                    cfg.get("models"),
                    cfg.get(
                        "training",
                        None,
                    ),
                    #input_dim,
                    input_shape,
                )
            )

            # --------------------------------------------------
            # 7. Train
            # --------------------------------------------------

            wrapper.fit(
                X_train_model,
                X_val_model,
            )

            # --------------------------------------------------
            # 8. Validation scores
            # --------------------------------------------------

            scores = (
                wrapper.get_scores(
                    X_val_model
                )
            )

            # --------------------------------------------------
            # 9. Thresholding
            # --------------------------------------------------

            thresholding = None

            if cfg.get("thresholding"):

                thresholding = (
                    Thresholding(
                        cfg.get(
                            "thresholding"
                        )
                    )
                )

                # Threshold is learned from
                # training scores
                train_scores = (
                    wrapper.get_scores(
                        X_train_model
                    )
                )

                thresholding.fit(
                    train_scores
                )

                threshold = (
                    thresholding.get_threshold()
                )

                predictions = (
                    wrapper.predict(
                        X_val_model,
                        threshold,
                    )
                )

            else:

                # Models such as Isolation Forest
                # use their native prediction mechanism
                predictions = (
                    wrapper.predict(
                        X_val_model
                    )
                )

            # --------------------------------------------------
            # 10. Sanity check
            # --------------------------------------------------

            print(
                "\nEvaluation shapes:"
            )

            print(
                "  scores:",
                len(scores)
            )

            print(
                "  y_val:",
                len(y_val_w)
            )

            print(
                "  predictions:",
                len(predictions)
            )

            assert len(scores) == len(y_val_w), (
                f"scores ({len(scores)}) != "
                f"y_val ({len(y_val_w)})"
            )

            assert len(predictions) == len(y_val_w), (
                f"predictions ({len(predictions)}) != "
                f"y_val ({len(y_val_w)})"
            )

            # --------------------------------------------------
            # 11. Evaluate
            # --------------------------------------------------

            metrics = (
                self.evaluator.evaluate(
                    scores=scores,
                    y_true=y_val_w,
                    predictions=predictions,
                )
            )

            # --------------------------------------------------
            # 12. Log run
            # --------------------------------------------------

            self.logger.log_run(
                cfg=cfg,
                run_type=run_type,
                metrics=metrics,
                history=wrapper.history,
                preprocessor=preprocessor,
                thresholding=thresholding,
                wrapper=(
                    wrapper
                    if metrics["auc"] > 0.7
                    else None
                ),
            )

        return metrics

"""
try:
    preprocessor.fit_transform(X_train)
except ValueError as e:
    raise optuna.TrialPruned(str(e))

idea to prune bad confis
"""


"""
AEEntry.prepare_input()
    -> (N, T, F) -> (N, T*F)

IsoForestEntry.prepare_input()
    -> (N, T, F) -> (N, T*F)

TransformerEntry.prepare_input()
    -> (N, T, F) -> unchanged
"""