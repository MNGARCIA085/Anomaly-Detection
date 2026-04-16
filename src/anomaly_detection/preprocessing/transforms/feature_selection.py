


from anomaly_detection.preprocessing.base import Transform



class FeatureSelector(Transform):

    def __init__(self, indices):
        self.indices = indices

    def transform(self, X):
        return X[:, self.indices]

    # each class defines whats worth saving
    def get_artifacts(self):
       return {"indices": self.indices}



    @classmethod
    def from_artifacts(cls, artifacts):
        return cls(indices=artifacts["indices"])