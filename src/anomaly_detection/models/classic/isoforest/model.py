


# dsp. que herede
class IsoWrapper:

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







