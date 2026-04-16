from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Any, Optional, Dict, Tuple

#-----------------------------------------------------------
# CONFIGURATION OBJECTS (Data-only schemas)
#-----------------------------------------------------------

@dataclass
class PhaseConfig:
    lr: float = 1e-4
    epochs: int = 2
    unfreeze_layers: Optional[int] = None

@dataclass
class TrainingConfig:
    strategy: str = "standard"  # e.g., "standard", "transfer_learning", "sklearn"
    backbone_name: Optional[str] = "backbone"
    metrics: List[Any] = field(default_factory=list)
    loss: Any = "mse"
    phase1: PhaseConfig = field(default_factory=PhaseConfig)
    phase2: PhaseConfig = field(
        default_factory=lambda: PhaseConfig(lr=1e-5, epochs=1, unfreeze_layers=1)
    )

#-----------------------------------------------------------
# TRAINER STRATEGIES (The "How" of training)
#-----------------------------------------------------------

class BaseTrainer(ABC):
    def __init__(self, model: Any, cfg: TrainingConfig):
        self.model = model
        self.cfg = cfg

    @abstractmethod
    def train(self, train_gen: Any, val_gen: Any, callbacks: List[Any]) -> Dict[str, Any]:
        """Executes the training logic and returns a results dictionary."""
        pass

    def _compile_model(self, lr: float):
        # Placeholder for tf.keras.optimizers.Adam(lr) etc.
        self.model.compile(optimizer="adam", loss=self.cfg.loss, metrics=self.cfg.metrics)

class StandardTrainer(BaseTrainer):
    def train(self, train_gen, val_gen, callbacks) -> Dict[str, Any]:
        self._compile_model(lr=self.cfg.phase1.lr)
        history = self.model.fit(
            train_gen, 
            validation_data=val_gen, 
            epochs=self.cfg.phase1.epochs, 
            callbacks=callbacks
        )
        return {"history": history.history, "type": "standard"}

class TransferLearningTrainer(BaseTrainer):
    def __init__(self, model, cfg):
        super().__init__(model, cfg)
        # Identify the specific sub-module for freezing logic
        self.backbone = model.get_layer(cfg.backbone_name)

    def _set_backbone_trainable(self, trainable: bool, n_last: Optional[int] = None):
        if n_last is None:
            for layer in self.backbone.layers:
                layer.trainable = trainable
        else:
            # Unfreeze only the top N layers
            for layer in self.backbone.layers[:-n_last]:
                layer.trainable = False
            for layer in self.backbone.layers[-n_last:]:
                layer.trainable = True

    def train(self, train_gen, val_gen, callbacks) -> Dict[str, Any]:
        # Phase 1: Feature Extraction
        self._set_backbone_trainable(False)
        self._compile_model(lr=self.cfg.phase1.lr)
        h1 = self.model.fit(train_gen, validation_data=val_gen, epochs=self.cfg.phase1.epochs, callbacks=callbacks)

        # Phase 2: Fine Tuning
        self._set_backbone_trainable(True, n_last=self.cfg.phase2.unfreeze_layers)
        self._compile_model(lr=self.cfg.phase2.lr)
        h2 = self.model.fit(train_gen, validation_data=val_gen, epochs=self.cfg.phase2.epochs, callbacks=callbacks)

        return {"phase1": h1.history, "phase2": h2.history, "type": "transfer_learning"}

class SklearnTrainer(BaseTrainer):
    """Bridge for models that don't use generators or backbones."""
    def train(self, train_gen, val_gen, callbacks) -> Dict[str, Any]:
        # Logic to extract X_train from generator if needed
        X_train, _ = next(train_gen) 
        self.model.fit(X_train)
        return {"history": None, "type": "sklearn"}

#-----------------------------------------------------------
# MODEL WRAPPER & FACTORY (The "Glue")
#-----------------------------------------------------------

class AnomalyModelWrapper:
    """The unified interface the Experiment class interacts with."""
    def __init__(self, architecture: Any, trainer: BaseTrainer):
        self.architecture = architecture
        self.trainer = trainer

    def fit(self, train_data, val_data, callbacks=None):
        return self.trainer.train(train_data, val_data, callbacks or [])

    def get_scores(self, X):
        # Logic for reconstruction error or decision function
        pass

class TrainerFactory:
    @staticmethod
    def get_trainer(model: Any, cfg: TrainingConfig) -> BaseTrainer:
        if cfg.strategy == "transfer_learning":
            return TransferLearningTrainer(model, cfg)
        if cfg.strategy == "standard":
            return StandardTrainer(model, cfg)
        if cfg.strategy == "sklearn":
            return SklearnTrainer(model, cfg)
        raise ValueError(f"Unknown strategy: {cfg.strategy}")

class ModelFactory:
    @staticmethod
    def create_anomaly_model(model_type: str, model_params: dict, train_cfg: TrainingConfig) -> AnomalyModelWrapper:
        # 1. Build the raw architecture (NN or Sklearn)
        # In a real project, this would use a registry of architectures
        if model_type == "autoencoder":
            architecture = build_ae_architecture(**model_params)
        elif model_type == "iso_forest":
            architecture = build_forest_architecture(**model_params)
            
        # 2. Assign the trainer strategy
        trainer = TrainerFactory.get_trainer(architecture, train_cfg)
        
        # 3. Return the interface-compatible wrapper
        return AnomalyModelWrapper(architecture, trainer)