

from .trainer import NNTrainer
from anomaly_detection.models.nnets.vae.trainer import VAETrainer


TRAINER_REGISTRY = {
    "default": NNTrainer,
    "vae": VAETrainer,
    #"transfer": TransferTrainer
}