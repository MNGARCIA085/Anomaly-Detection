from ...base_model import AnomalyWrapper

from anomaly_detection.training.schemas import TrainingHistory

from ...persistence.sklearn import save_sklearn_model, load_sklearn_model



class IsoWrapper(AnomalyWrapper):

    def __init__(self, model):
        self.model = model


    def fit(
        self,
        X,
        y=None
    ):
        self.model.fit(X)
        return self


    def get_scores(self, X):
        return -self.model.decision_function(X)


    # property history; not really needed here, included just for an uniform interface
    @property
    def history(self):
        return TrainingHistory()


    # later, for unform inference
    @property
    def input_dim(self):
        return self.model.n_features_in_ # number of features stored by the fitted IsolationForest



    # save & load
    def save(self, path):

        save_sklearn_model(
            self.model,
            path
        )


    @classmethod
    def load(cls, path):

        model = load_sklearn_model(
            path
        )

        return cls(
            model=model
        )


