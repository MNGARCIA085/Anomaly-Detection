# src/core/factory.py


"""
from src.models import MODEL_REGISTRY


# maybe, and probably later
# model class + training class

class ModelFactory:
    @staticmethod
    def create(model_type, model_cfg):
        model_class = MODEL_REGISTRY.get(model_type)
        return model_class(model_cfg)
"""



from anomaly_detection.models.isoforest import IsolationForestModel



class ModelFactory:
    @staticmethod
    def create(model_type, model_cfg, runtime_params):
        input_dim = runtime_params["input_dim"]
        
        if model_type == "autoencoder":
            pass

            """
            # 1. Build Architecture
            architecture = AE(input_dim=input_dim, layers=cfg.layers)
            
            # 2. Handle Transfer Learning logic
            if cfg.get("pretrained_path"):
                architecture.load_state_dict(torch.load(cfg.pretrained_path))
                if cfg.freeze_encoder:
                    # Logic to freeze layers
                    pass
            
            # 3. Choose Trainer
            trainer = DeepLearningTrainer(epochs=cfg.epochs, lr=cfg.lr)
            
            # 4. Return the Wrapper
            return AutoencoderModel(architecture, trainer)
            """

        if model_type == "isoforest":
            #architecture = IsolationForest(**cfg.params)
            #trainer = SklearnTrainer()
            trainer = None
            return IsolationForestModel(model_cfg, trainer)








"""
Correct ownership:
Factory owns the decision
Wrapper owns the usage
Experiment doesn’t care
"""

"""
When would a TrainerFactory make sense?

Only if you reach something like:

Same model, multiple trainers:
AE + standard trainer
AE + contrastive trainer
AE + distributed trainer

Trainer becomes configurable like:

trainer: "ddp"
logging: "wandb"
early_stopping: true

At that point, yes:
👉 You split it.
"""