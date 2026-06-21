
# later import all
from anomaly_detection.new.ae import AEEntry
from anomaly_detection.new.iso import IsoEntry


from anomaly_detection.new.registry import MODEL_REGISTRY


class Experiment:

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

        # val scores
        scores = (
            wrapper.get_scores(
                X_val_p
            )
        )


        print(scores)

        # evaluator????


        # logger ?????

        return {
            "score": -float(
                scores.mean()
            )
        }