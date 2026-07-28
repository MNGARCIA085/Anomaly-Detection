

from .trainer import NNTrainer


TRAINER_REGISTRY = {
    "default": NNTrainer,
    #"vae": VAETrainer,
    #"transfer": TransferTrainer
}