from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torch.optim as optim

from .dataset import AnomalyDataset
from .schemas import TrainState, TrainingConfig




# later -> optim... as DI -> it seems to work well
class BaseTrainer:

    def __init__(self, cfg: TrainingConfig):
        self.cfg = cfg
        self.callbacks = cfg.callbacks or []

    def _call_callbacks(self, hook, state):
        for cb in self.callbacks:
            getattr(cb, hook, lambda x: None)(state)

    def fit(self, model, X_train, X_val=None): # train
        device = self.cfg.device
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.lr)
        criterion = nn.MSELoss()

        # ---- DataLoaders ----
        train_loader = DataLoader(
            AnomalyDataset(X_train),
            batch_size=self.cfg.batch_size,
            shuffle=self.cfg.shuffle,
            num_workers=self.cfg.num_workers
        )

        val_loader = None
        if X_val is not None:
            val_loader = DataLoader(
                AnomalyDataset(X_val),
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=self.cfg.num_workers
            )

        state = TrainState(model=model)
        self._call_callbacks("on_train_start", state)

        for epoch in range(self.cfg.epochs):
            state.epoch = epoch
            epoch_loss = 0.0

            self._call_callbacks("on_epoch_start", state)

            model.train()
            for batch in train_loader:
                batch = batch.to(device)

                optimizer.zero_grad()
                recon = model(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch.size(0)

            epoch_loss /= len(train_loader.dataset)
            state.train_loss = epoch_loss

            # ---- Validation ----
            if val_loader is not None:
                model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(device)
                        recon = model(batch)
                        loss = criterion(recon, batch)
                        val_loss += loss.item() * batch.size(0)

                val_loss /= len(val_loader.dataset)
                state.val_loss = val_loss

                model.train()

            self._call_callbacks("on_epoch_end", state)

            if state.stop_training:
                break

        self._call_callbacks("on_train_end", state)

        return model