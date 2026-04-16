


from anomaly_detection.preprocessing.base import Transform



class ScalerTransform(Transform):

    def __init__(self, scaler_cls):
        self.scaler = scaler_cls()

    def fit(self, X):
        self.scaler.fit(X)
        return self

    def transform(self, X):
        return self.scaler.transform(X)

    def get_artifacts(self):
    	return self.scaler

    """
    def get_artifacts(self):
	    return {
	        "type": "scaler",
	        "object": self.scaler
	    }
    """

    # to restore artifacts
    @classmethod
    def from_artifacts(cls, artifacts):
        return cls(scaler=artifacts["scaler"])


# methoids to reconstruict later:https://chatgpt.com/c/69e137bc-6288-83e9-8412-9bfb3d865a07