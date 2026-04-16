

from abc import ABC



class Transform(ABC):

    def fit(self, X):
        return self

    def transform(self, X):
        return X

    # each class defines what worths saving
    def get_artifacts(self):
        raise NotImplementedError(f"{self.__class__.__name__} must implement get_artifacts")

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    


"""
fit() → learn parameters (mean, PCA components, etc.)
transform() → apply them
"""