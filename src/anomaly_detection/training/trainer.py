from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torch.optim as optim

from .dataset import AnomalyDataset
from .schemas import TrainState, TrainingConfig, TrainingHistory





class NNTrainer:


    def __init__(self, cfg: TrainingConfig):
        self.cfg = cfg
        self.callbacks = cfg.callbacks or []
        self.history = None
        self.optimizer = cfg.optimizer
        self.criterion = cfg.loss


    #--------build dataloader-----------#
    def build_dataloader(self, X, shuffle):
        return DataLoader(
            AnomalyDataset(X),
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
        )


    #------------Callbacks---------------#
    def _call_callbacks(self, hook, state):
        for cb in self.callbacks:
            getattr(cb, hook, lambda x: None)(state)


    #---------Training step--------------#
    #... later some models inherits tis trainer and can overwrite this
    def training_step(
        self,
        model,
        batch,
        criterion,
    ):
        recon = model(batch)
        return criterion(recon, batch)

    #---------Train epoch----------------#
    def train_epoch(
        self,
        model,
        loader,
        optimizer,
        criterion,
    ):
        model.train()

        total_loss = 0.0

        for batch in loader:
            batch = batch.to(self.cfg.device)

            optimizer.zero_grad()

            loss = self.training_step(
                model,
                batch,
                criterion,
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.size(0)

        return total_loss / len(loader.dataset)



    #-----------Validation epoch----------------#
    def validate(
        self,
        model,
        loader,
        criterion,
    ):
        model.eval()

        total_loss = 0.0

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.cfg.device)

                loss = self.training_step(
                    model,
                    batch,
                    criterion,
                )

                total_loss += loss.item() * batch.size(0)

        return total_loss / len(loader.dataset)


    #-----------------------Fit-----------------------#
    def fit(self, model, X_train, X_val=None):
        model.to(self.cfg.device)


        train_loader = self.build_dataloader(X_train, shuffle=self.cfg.shuffle)
        val_loader = (
            self.build_dataloader(X_val, shuffle=False)
            if X_val is not None
            else None
        )

        history = TrainingHistory()
        state = TrainState(model=model)

        self._call_callbacks("on_train_start", state)

        for epoch in range(self.cfg.epochs):
            state.epoch = epoch

            self._call_callbacks("on_epoch_start", state)

            train_loss = self.train_epoch(
                model,
                train_loader,
                self.optimizer,
                self.criterion,
            )

            state.train_loss = train_loss

            history.append("train_loss", train_loss)

            if val_loader is not None:
                val_loss = self.validate(
                    model,
                    val_loader,
                    self.criterion,
                )

                state.val_loss = val_loss
                history.append("val_loss", val_loss)

            self._call_callbacks("on_epoch_end", state)

            if state.stop_training:
                break

        self._call_callbacks("on_train_end", state)

        self.history = history

        return model





"""

vaetrainer(NNTrainer):

    # overwrite and add what i need
    def training_step....


VAE

def training_step(self, model, batch, criterion):
    recon, mu, logvar = model(batch)
    return criterion(recon, batch, mu, logvar)


dEEP svdd
def training_step(self, model, batch, criterion):
    z = model(batch)
    return criterion(z)
"""









"""
in YAML

The key is: optimizer.params is an opaque dictionary owned by the optimizer factory. 
Your generic infrastructure should not know that Adam has betas or SGD has momentum
"""


"""
for several models maybe use one ineriatance level

def training_step(self, model, batch, criterion):
    recon = model(batch)
    return criterion(recon, batch)


def training_step(self, model, batch, criterion):
    recon, mu, logvar = model(batch)

    recon_loss = criterion(recon, batch)
    kl = ...

    return recon_loss + beta * kl

def validation_step(self, model, batch, criterion):
    recon = model(batch)
    return criterion(recon, batch)

"""
