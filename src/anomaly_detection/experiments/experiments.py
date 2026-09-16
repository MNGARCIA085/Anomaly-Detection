import anomaly_detection.models.register_models # to trigger registration
from anomaly_detection.models.registry import MODEL_REGISTRY
from anomaly_detection.evaluation.evaluator import Evaluator
from anomaly_detection.infra.logging.null_logger import NullLogger
from anomaly_detection.thresholding.registry import create_threshold_strategy
from anomaly_detection.thresholding.thresholding import Thresholding
from anomaly_detection.data.windowing import Windowing



class Experiment:

    def __init__(
        self,
        model_type,
        evaluator,
        logger=None,
    ):
        self.model_type = model_type
        self.evaluator = evaluator
        self.logger = logger or NullLogger()

    # --------------------------------------------------
    # Data preparation
    # --------------------------------------------------

    def _prepare_data(
        self,
        cfg,
        entry,
        X_train,
        X_val,
        y_val,
    ):


        # point-wise prep -> windowing -> win. level prep (opt) -> model adapt input


        # Preprocessing
        preprocessor = entry.build_preprocessor(
            cfg.get("prep")
        )

        # point-wise prep (scaling.....)
        X_train_p = preprocessor.fit_transform(
            X_train
        )

        X_val_p = preprocessor.transform(
            X_val
        )

        # Windowing
        windowing_cfg = (
            cfg.get("data", {})
               .get("windowing", {})
        )

        window_size = windowing_cfg.get(
            "size",
            8,
        )

        windowing = Windowing(
            window_size
        )

        X_train_w = windowing.transform(
            X_train_p
        )

        X_val_w, y_val_w = (
            windowing.transform_with_labels(
                X_val_p,
                y_val,
            )
        )

        # Optional temporal preprocessing (window-level prep)
        temporal_cfg = (
            cfg.get("prep", {})
               .get("temporal")
        )

        temporal_preprocessor = None

        if temporal_cfg:

            temporal_preprocessor = (
                entry.build_temporal_preprocessor(
                    temporal_cfg
                )
            )

            X_train_w = (
                temporal_preprocessor.fit_transform(
                    X_train_w
                )
            )

            X_val_w = (
                temporal_preprocessor.transform(
                    X_val_w
                )
            )

        # Model-specific representation
        X_train_model = entry.adapt_input(
            X_train_w
        )

        X_val_model = entry.adapt_input(
            X_val_w
        )

        artifacts = {
            "preprocessor": preprocessor,
            "windowing": windowing,
            "temporal_preprocessor": (
                temporal_preprocessor
            ),
            "y_val_w": y_val_w,
        }

        return (
            X_train_model,
            X_val_model,
            artifacts,
        )

    # --------------------------------------------------
    # Model construction
    # --------------------------------------------------

    def _build_model(
        self,
        cfg,
        entry,
        X_train_model,
    ):
        input_shape = X_train_model.shape

        return entry.build(
            cfg.get("models"),
            cfg.get("training", None),
            input_shape,
        )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    def _train(
        self,
        wrapper,
        X_train_model,
        X_val_model,
    ):
        wrapper.fit(
            X_train_model,
            X_val_model,
        )

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    def _get_predictions(
        self,
        cfg,
        wrapper,
        train_scores,
        val_scores,
        y_val_w,
        X_val_model,
    ):
        thresholding_cfg = cfg.get(
            "thresholding"
        )

        # Models with native prediction
        if not thresholding_cfg:

            predictions = wrapper.predict(
                X_val_model
            )

            return predictions, None

        # Models using explicit thresholding
        thresholding = Thresholding(
            thresholding_cfg
        )

        thresholding.fit(
            train_scores=train_scores,
            val_scores=val_scores,
            y_val=y_val_w,
        )

        threshold = (
            thresholding.get_threshold()
        )

        predictions = wrapper.predict(
            X_val_model,
            threshold,
        )

        return predictions, thresholding

    # --------------------------------------------------
    # Logging
    # --------------------------------------------------

    def _log_run(
        self,
        cfg,
        run_type,
        metrics,
        wrapper,
        artifacts,
        thresholding,
    ):
        self.logger.log_run(
            cfg=cfg,
            run_type=run_type,
            metrics=metrics,
            history=wrapper.history,
            preprocessor=(
                artifacts["preprocessor"]
            ),
            windowing=(
                artifacts["windowing"]
            ),
            temporal_preprocessor=(
                artifacts["temporal_preprocessor"]
            ),
            thresholding=thresholding,
            wrapper=wrapper,
        )

    # --------------------------------------------------
    # Main experiment pipeline
    # --------------------------------------------------

    def run(
        self,
        cfg,
        X_train,
        X_val,
        y_val=None,
        run_type="train",
    ):

        with self.logger.start_run(
            run_name=self.model_type
        ):

            # 1. Build model entry
            entry = MODEL_REGISTRY[
                self.model_type
            ]()

            # 2. Prepare data (pointwise prep + wind. + win. level prep + adapt input)
            (
                X_train_model,
                X_val_model,
                artifacts,
            ) = self._prepare_data(
                cfg=cfg,
                entry=entry,
                X_train=X_train,
                X_val=X_val,
                y_val=y_val,
            )

            # 3. Build model
            wrapper = self._build_model(
                cfg=cfg,
                entry=entry,
                X_train_model=X_train_model,
            )

            # 4. Train
            self._train(
                wrapper=wrapper,
                X_train_model=X_train_model,
                X_val_model=X_val_model,
            )

            # 5. Calculate scores ONCE
            train_scores = wrapper.get_scores(
                X_train_model
            )

            val_scores = wrapper.get_scores(
                X_val_model
            )

            # 6. Predictions / thresholding
            (
                predictions,
                thresholding,
            ) = self._get_predictions(
                cfg=cfg,
                wrapper=wrapper,
                train_scores=train_scores,
                val_scores=val_scores,
                y_val_w=artifacts["y_val_w"],
                X_val_model=X_val_model,
            )

            # 7. Evaluation
            metrics = self.evaluator.evaluate(
                scores=val_scores,
                y_true=artifacts["y_val_w"],
                predictions=predictions,
            )

            # 8. Logging
            self._log_run(
                cfg=cfg,
                run_type=run_type,
                metrics=metrics,
                wrapper=wrapper,
                artifacts=artifacts,
                thresholding=thresholding,
            )

        return metrics
