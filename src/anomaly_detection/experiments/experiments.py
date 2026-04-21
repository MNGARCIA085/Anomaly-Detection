
from anomaly_detection.models.factory import ModelFactory


class Experimentv0:
    def __init__(self, preprocessor, threshold_selector, evaluator, logger=None):
        self.preprocessor = preprocessor
        self.threshold_selector = threshold_selector
        self.evaluator = evaluator
        self.logger = logger

    def run(self, model_wrapper, X_train, X_val, y_val):
        """
        Runs the full pipeline using a pre-built model_wrapper from the Factory. (model_wrap has the model + trainer)
        """
        # 1. Data Prep
        X_train_p, X_val_p = self.preprocessor.fit_transform_split(X_train, X_val)

        # 2. Fit (The wrapper handles whether it's Sklearn or a 2-phase TL)
        training_results = model_wrapper.fit(X_train_p, X_val_p)

        # 3. Score & Threshold
        scores_val = model_wrapper.get_scores(X_val_p)
        threshold = self.threshold_selector.from_pr_curve(y_val, scores_val)

        # 4. Evaluation
        metrics = self.evaluator.evaluate(y_val, scores_val, threshold)

        # 5. Logging (Optional)
        if self.logger:
            self.logger.log_metrics(metrics)
            self.logger.log_params(model_wrapper.trainer.cfg.__dict__)
            self.logger.log_results(training_results)

        return metrics, threshold






class Experiment:

    def __init__(self, preprocessor, threshold_selector, evaluator, model_type, model_cfg, logger=None):
        self.model_type = model_type
        self.model_cfg = model_cfg
        self.preprocessor = preprocessor
        self.threshold_selector = threshold_selector
        self.evaluator = evaluator
        self.logger = logger

    def run(self, X_train, X_val, y_val):

        X_train_p, X_val_p = self.preprocessor.fit_transform_split(X_train, X_val)


        # Build the model (High-level orchestration)
        # Input dim is dynamic and only needed for nns
        # We pass the type and the params. The Factory handles the 'input_dim' logic.
        model_factory = ModelFactory()
        self.model = model_factory.create( # later also pass training config (and include in there the staretgy)!!!!
            model_type=self.model_type,
            model_cfg=self.model_cfg,
            runtime_params={"input_dim": X_train_p.shape[1]}
        )
        # prev code is a wrapper, returns isntnacniated model + trainer


        self.model.fit(X_train_p, X_val_p)

        scores_val = self.model.get_scores(X_val_p)

        threshold = self.threshold_selector.from_pr_curve(y_val, scores_val)

        metrics = self.evaluator.evaluate(y_val, scores_val, threshold)

        
        # 🔥 log artifacts
        if self.logger:
            artifacts = self.preprocessor.get_artifacts()
            self.logger.log_artifacts(artifacts)
        

        return metrics, threshold

