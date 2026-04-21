

class Experiment:

    def __init__(self, preprocessor, threshold_selector, evaluator, logger=None):
        self.preprocessor = preprocessor
        self.threshold_selector = threshold_selector
        self.evaluator = evaluator
        self.logger = logger

    def run(self, model_builder, X_train, X_val, y_val):

        # 1. Preprocessing
        X_train_p, X_val_p = self.preprocessor.fit_transform_split(X_train, X_val)

         # 2. Runtime params
        runtime_params = {
            "input_dim": X_train_p.shape[1]
        }

        # 3. Wrapper (model + trainer)
        model_wrapper = model_builder(runtime_params) 

        # 4. Train
        model_wrapper.fit(X_train_p, X_val_p)

        # 5. Score + threshold
        scores_val = model_wrapper.get_scores(X_val_p)
        threshold = self.threshold_selector.from_pr_curve(y_val, scores_val)

        # 6. Evaluate
        metrics = self.evaluator.evaluate(y_val, scores_val, threshold)

        
        # 🔥 log artifacts
        if self.logger:
            artifacts = self.preprocessor.get_artifacts()
            self.logger.log_artifacts(artifacts)
        

        return metrics, threshold

