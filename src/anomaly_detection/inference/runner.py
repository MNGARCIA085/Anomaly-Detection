


class InferenceRunner:

    def __init__(
        self,
        prep,
        windowing,
        entry,
        wrapper,
        thresholding=None,
    ):
        self.prep = prep
        self.windowing = windowing
        self.entry = entry
        self.wrapper = wrapper
        self.thresholding = thresholding

    def predict(self, X):

        X_p = self.prep.transform(X)

        X_w = self.windowing.transform(X_p)

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