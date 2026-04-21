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



from anomaly_detection.models.isoforest import IsolationForestModel, build_forest


from anomaly_detection.models.ae import AE, AETrainer, AutoencoderModel





class ModelFactory:
    @staticmethod
    def create(model_type, model_cfg, runtime_params):
        

        #input_dim = runtime_params["input_dim"] # error if i dont pass it, put guards
        input_dim = runtime_params.get("input_dim", 11)


        
        if model_type == "ae":

            #input_dim = 8
            model_cfg['input_dim'] = input_dim # see later if this is clean!!!; maybe a merge is better
            #print(model_cfg)

            # add input_dim to config!!! OR update default value

            # build model
            model = AE(model_cfg)

            # choose trainer
            trainer = AETrainer() # quick test, adapt later, pass config, see TL
            
            # return the warpper
            return AutoencoderModel(model, trainer)


        if model_type == "isoforest":
            model = build_forest(model_cfg) # or **cfg.params
            print(model)
            trainer = None
            return IsolationForestModel(model, trainer) # change!!!; pass inst. model




# create a builder



"""
Correct ownership:
Factory owns the decision (but also creates the model)
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