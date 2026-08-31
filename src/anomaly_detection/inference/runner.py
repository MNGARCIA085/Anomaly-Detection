



class InferenceRunner:

    def __init__(
        self,
        prep,
        windowing,
        entry,
        wrapper,
        temporal_prep=None,
        thresholding=None,
    ):
        self.prep = prep
        self.temporal_prep = temporal_prep
        self.windowing = windowing
        self.entry = entry
        self.wrapper = wrapper
        self.thresholding = thresholding

    def _transform(self, X):

        X_p = self.prep.transform(X)

        X_w = self.windowing.transform(X_p)

        if self.temporal_prep is not None:
            X_w = self.temporal_prep.transform(X_w)

        return self.entry.adapt_input(X_w)


    def _transform_with_labels(self, X, y):

        X_p = self.prep.transform(X)

        X_w, y_w = self.windowing.transform_with_labels(
            X_p,
            y,
        )

        if self.temporal_prep is not None:
            X_w = self.temporal_prep.transform(X_w)

        X_model = self.entry.adapt_input(X_w)

        return X_model, y_w





    def score(self, X):

        X_model = self._transform(X)

        return self.wrapper.get_scores(X_model)





    def predict(self, X):

        X_model = self._transform(X)

        if self.thresholding is not None:

            threshold = (
                self.thresholding.get_threshold()
            )

            return self.wrapper.predict(
                X_model,
                threshold,
            )

        return self.wrapper.predict(X_model)




    def predict_with_labels(self, X, y):

        X_model, y_w = self._transform_with_labels(
            X,
            y,
        )

        scores = self.wrapper.get_scores(X_model)

        if self.thresholding is not None:

            threshold = (
                self.thresholding.get_threshold()
            )

            predictions = self.wrapper.predict(
                X_model,
                threshold,
            )

        else:

            predictions = self.wrapper.predict(
                X_model
            )

        return scores, y_w, predictions













#------------------
class InferenceRunnerv0:

    def __init__(
        self,
        prep,
        windowing,
        entry,
        wrapper,
        temporal_prep=None,
        thresholding=None,
    ):
        self.prep = prep
        self.temporal_prep = temporal_prep
        self.windowing = windowing
        self.entry = entry
        self.wrapper = wrapper
        self.thresholding = thresholding


    def score(self, X):

        X_p = self.prep.transform(X)
        X_w = self.windowing.transform(X_p)

        if self.temporal_prep is not None:
            X_w = self.temporal_prep.transform(X_w)

        X_model = self.entry.adapt_input(X_w)

        return self.wrapper.get_scores(X_model)



    def predict(self, X):

        scores = self.score(X)

        if self.thresholding is not None:
            threshold = self.thresholding.get_threshold()
            return self.thresholding.predict(scores)

        return scores


    """
    def predict(self, X):

        X_p = self.prep.transform(X)

        X_w = self.windowing.transform(X_p)

        if self.temporal_prep is not None: # window-level!!!

            X_w = self.temporal_prep.transform(X_w)

        X_model = self.entry.adapt_input(X_w)

        if self.thresholding is not None:

            threshold = (
                self.thresholding.get_threshold()
            )

            return self.wrapper.predict(
                X_model,
                threshold,
            )

        return self.wrapper.predict(X_model)
    """



"""
Raw X
 ↓
prep.transform()
 ↓
windowing.transform()
 ↓
temporal_prep.transform()   # optional
 ↓
entry.adapt_input()
 ↓
wrapper.predict()
"""