
# later import all
from anomaly_detection.new_2.ae import AEEntry
from anomaly_detection.new_2.iso import IsoEntry


from anomaly_detection.new_2.registry import MODEL_REGISTRY



from .evaluator import Evaluator




#----------
class Experiment:

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
        ]

        preprocessor = (
            entry.build_preprocessor(cfg)
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
                cfg,
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

        print(y_val)

        evaluation = (
            self.evaluator.evaluate(
                scores=scores,
                y_true=y_val,
                X=X_val_p
            )
        )

        print('dsfdsfs', evaluation)

        return evaluation






#------------------

class Experimentv0:

    def __init__(
        self,
        model_type
    ):
        self.model_type = model_type

    def run(
        self,
        cfg,
        X_train,
        X_val
    ):

        entry = MODEL_REGISTRY[
            self.model_type
        ]

        preprocessor = (
            entry.build_preprocessor(cfg)
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


        print('vamos')


        print(cfg)

        wrapper = (
            entry.build(
                cfg,
                input_dim
            )
        )

        print(wrapper)

        wrapper.fit( # fit
            X_train_p,
            X_val_p
        )

        # val scores
        scores = (
            wrapper.get_scores(
                X_val_p
            )
        )



        return {
            "score": -float(
                scores.mean()
            )
        }