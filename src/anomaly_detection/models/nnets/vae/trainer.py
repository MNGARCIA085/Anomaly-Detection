import torch
import torch.nn.functional as F

from anomaly_detection.training.trainer import NNTrainer


class VAETrainer(NNTrainer):

    def training_step(
        self,
        model,
        batch,
        criterion=None,
    ):

        recon, mu, logvar = model(batch)

        recon_loss = F.mse_loss(
            recon,
            batch,
            reduction="mean"
        )

        kl_loss = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )

        loss = (
            recon_loss
            + model.config.beta * kl_loss
        )

        return loss