

from .trainer import BaseTrainer


TRAINER_REGISTRY = {
    "base": BaseTrainer,
    #"vae": VAETrainer,
    #"transfer": TransferTrainer
}