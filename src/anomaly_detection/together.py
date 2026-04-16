class Experiment:
    def __init__(self, preprocessor, threshold_selector, evaluator, logger=None):
        self.preprocessor = preprocessor
        self.threshold_selector = threshold_selector
        self.evaluator = evaluator
        self.logger = logger

    def run(self, model_wrapper, X_train, X_val, y_val):
        """
        Runs the full pipeline using a pre-built model_wrapper from the Factory.
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




def main():
    # --- 1. SETUP DATA ---
    dm = DataModule(train_path="data/train.npy", val_path="data/val.npy", y_val_path="data/y_val.npy")
    X_train, X_val, y_val = dm.load()

    # --- 2. SETUP COMPONENTS ---
    prep = PreprocessingPipeline([ScalerTransform(StandardScaler)])
    evaluator = Evaluator()
    selector = ThresholdSelector()
    
    # Initialize the experiment runner
    exp = Experiment(prep, selector, evaluator, logger=MLflowLogger())

    # --- 3. CONFIGURATION (In reality, this comes from Hydra/YAML) ---
    # Example for a Transfer Learning AE
    train_cfg = TrainingConfig(
        strategy="transfer_learning",
        backbone_name="encoder_v1",
        phase1=PhaseConfig(lr=1e-3, epochs=10),
        phase2=PhaseConfig(lr=1e-5, epochs=5, unfreeze_layers=2)
    )
    
    model_params = {"layers": [64, 32, 16], "dropout": 0.2}

    # --- 4. EXECUTION ---
    # Use the Factory to build the ready-to-use wrapper
    model_wrapper = ModelFactory.create_anomaly_model(
        model_type="autoencoder",
        model_params=model_params,
        train_cfg=train_cfg
    )

    # Run the experiment
    print("Starting Experiment...")
    final_metrics, final_threshold = exp.run(model_wrapper, X_train, X_val, y_val)
    
    print(f"Final F1 Score: {final_metrics['f1']:.4f}")

# --- OPTIONAL: TUNING MODE ---
def run_tuning():
    # ... load data ...
    tuner = OptunaTuner(model_factory=ModelFactory(), experiment_runner=exp)
    
    search_space = {
        "lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        "unfreeze_layers": {"type": "int", "low": 1, "high": 5}
    }
    
    best_params = tuner.tune(X_train, X_val, y_val, search_space)
    print(f"Best Hyperparameters: {best_params}")

if __name__ == "__main__":
    main()