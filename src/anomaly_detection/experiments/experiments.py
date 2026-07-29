from anomaly_detection.models.nnets.ae.entry import AEEntry
from anomaly_detection.models.classic.isoforest.entry import IsoEntry
from anomaly_detection.models.registry import MODEL_REGISTRY

from anomaly_detection.evaluation.evaluator import Evaluator











from anomaly_detection.infra.logger import flatten_dict, MLFlowLogger




class Experiment:

    def __init__(
        self,
        model_type,
        evaluator,
        logger=MLFlowLogger(),
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

            # log params from my config
            self.logger.log_params(
                flatten_dict(cfg)
            )




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


            evaluation = (
                self.evaluator.evaluate(
                    scores=scores,
                    y_true=y_val,
                    X=X_val_p
                )
            )

        return evaluation












#--------------------------OK---------------------
class Experimentv0:

    def __init__(
        self,
        model_type,
        evaluator
    ):
        self.model_type = model_type
        self.evaluator = evaluator

    def run(
        self,
        cfg,
        X_train,
        X_val,
        y_val=None
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

        input_dim = (
            X_train_p.shape[1]
        )


        wrapper = (
            entry.build(
                cfg.get('models'),
                cfg.get('training', None), # None for isoforests......
                input_dim
            )
        )



        # log params from my config


        wrapper.fit(
            X_train_p,
            X_val_p
        )

        scores = (
            wrapper.get_scores(
                X_val_p
            )
        )


        evaluation = (
            self.evaluator.evaluate(
                scores=scores,
                y_true=y_val,
                X=X_val_p
            )
        )

        return evaluation



# logging: https://chatgpt.com/c/6a6a64a4-6a7c-83e9-8abe-b720fb6e5351