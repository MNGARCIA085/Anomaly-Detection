import anomaly_detection.models.register_models # to trigger registration
from anomaly_detection.models.registry import MODEL_REGISTRY

from anomaly_detection.evaluation.evaluator import Evaluator

from anomaly_detection.infra.utils import flatten_dict
from anomaly_detection.infra.mlflow_logger import  MLFlowLogger




class Experiment:

    def __init__(
        self,
        model_type,
        evaluator,
        logger=MLFlowLogger(), # later ill make it optional
    ):
        self.model_type = model_type
        self.evaluator = evaluator
        self.logger = logger

    def run(
        self,
        cfg,
        X_train,
        X_val,
        y_val=None
    ):

        with self.logger.start_run(
            run_name=self.model_type
        ):

            # tags
            self.logger.log_tags(cfg.name)


            # log params from my config; maybe prefixes later
            self.logger.log_params(
                flatten_dict(cfg.get('prep'))
            )
            self.logger.log_params(
                flatten_dict(cfg.get('models'))
            )

            if cfg.get('training', None):
                self.logger.log_params(
                    flatten_dict(cfg.get('training'))
                )
            #--------------------------

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

            input_dim = (
                X_train_p.shape[1]
            )



            # prep -> save after fit!!!!
            path = self.logger.artifact_path("preprocessor.pkl")
            preprocessor.save(path)
            self.logger.log_artifact(
                path,
                artifact_path="preprocessing"
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


            # history
            history = wrapper.history

            # log:
            if wrapper.history.metrics:
                self.logger.log_training_history(wrapper.history)



            scores = (
                wrapper.get_scores(
                    X_val_p
                )
            )


            # is it the same for the transfomer!!!!
            evaluation = (
                self.evaluator.evaluate(
                    scores=scores,
                    y_true=y_val,
                    X=X_val_p
                )
            ) # actually returns metrics


            # log metrics
            self.logger.log_metrics(evaluation)
            #---------


            # save only "good" models
            if evaluation["auc"] > 0.9:

                path = self.logger.artifact_path("model")

                wrapper.save(path)

                self.logger.log_artifact(
                    path,
                    artifact_path="model"
                )

        return evaluation


