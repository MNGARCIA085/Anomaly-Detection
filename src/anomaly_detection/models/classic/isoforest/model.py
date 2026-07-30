from ...base_model import AnomalyWrapper


import joblib


from anomaly_detection.training.schemas import TrainingHistory


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


    #...
    def save(self, path):

        joblib.dump(
            self.model,
            path
        )


    # property history; not really needed here
    @property
    def history(self):
        return TrainingHistory()







