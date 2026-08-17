import anomaly_detection.models.register_models # to trigger registration
from anomaly_detection.models.registry import MODEL_REGISTRY

from anomaly_detection.evaluation.evaluator import Evaluator

#from anomaly_detection.infra.utils import flatten_dict
#from anomaly_detection.infra.mlflow_logger import  MLFlowLogger


from anomaly_detection.infra.null_logger import NullLogger

from anomaly_detection.thresholding.thresholding import create_threshold_strategy
from anomaly_detection.thresholding.thresholding import Thresholding


from anomaly_detection.data.windowing import Windowing



# woks without windowing!!!!


class Experiment:

    def __init__(
        self,
        model_type,
        evaluator,
        logger=None,
    ):
        self.model_type = model_type
        self.evaluator = evaluator
        self.logger = self.logger = logger or NullLogger()

    def run(
        self,
        cfg,
        X_train,
        X_val,
        y_val=None,
        run_type="train", # train or tune
    ):

        with self.logger.start_run(
            run_name=self.model_type
        ):



            entry = MODEL_REGISTRY[
                self.model_type
            ]()

            preprocessor = (
                entry.build_preprocessor(cfg.get('prep'))
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


            # ---- see this
            input_dim = (
                X_train_p.shape[1]
            )

            # wrapper
            wrapper = (
                entry.build(
                    cfg.get('models'),
                    cfg.get('training', None), # None for isoforests......
                    input_dim
                )
            )


            wrapper.fit(
                X_train_p,
                X_val_p
            )


            scores = (
                wrapper.get_scores(
                    X_val_p
                )
            )


            # predictions and threshold
            thresholding = None
            
            if cfg.get("thresholding"): # AEs, VAEs...
                thresholding = Thresholding(
                    cfg.get("thresholding")
                )

                train_scores = wrapper.get_scores(X_train_p)

                thresholding.fit(train_scores)

                threshold = (
                    thresholding.get_threshold()
                )

                predictions = wrapper.predict(
                    X_val_p,
                    threshold,
                )

            else: # isoforests
                predictions = wrapper.predict(
                    X_val_p
                )


            # compute metrics
            metrics = self.evaluator.evaluate(
                scores=scores,
                y_true=y_val,
                predictions=predictions,
            )


            # log run
            self.logger.log_run(
                cfg=cfg,
                run_type=run_type,
                metrics=metrics,
                history=wrapper.history,
                preprocessor=preprocessor, # already fit!
                thresholding=thresholding, # already fit
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