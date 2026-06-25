from sklearn.metrics import roc_auc_score


class Evaluator:

    def evaluate(
        self,
        scores,
        y_true=None,
        X=None
    ):
        result = {}

        result["mean_score"] = (
            scores.mean()
        )

        if y_true is not None:
            result["auc"] = (
                roc_auc_score(
                    y_true,
                    scores
                )
            )

        return result




# remerbe the sign, have it in mind!!!!