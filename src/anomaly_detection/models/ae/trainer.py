import torch
import torch.nn as nn
from anomaly_detection.models.ae.schemas import AETrainingConfig, TrainState



# callbacks interface
class Callback:
    def on_train_start(self, state): pass
    def on_epoch_start(self, state): pass
    def on_epoch_end(self, state): pass
    def on_train_end(self, state): pass


class PrintLossCallback(Callback):
    def on_epoch_end(self, state):
        print(f"Epoch {state.epoch} - Train Loss: {state.train_loss:.4f} - Val Loss: {state.val_loss:.4f}")


class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.best = float("inf")
        self.counter = 0

    def on_epoch_end(self, state):
        if state.val_loss is None:
            return

        if state.val_loss < self.best:
            self.best = state.val_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            print('ES triggreed')
            state.stop_training = True






# dataloader
import torch
from torch.utils.data import Dataset

class AEDataset(Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]




#-----------------------------TRAINER--------------------------#

from torch.utils.data import DataLoader
import torch.nn as nn
import torch






# -----------------------
# Trainer
# -----------------------
# later -> optim... as DI -> it seems to work well
class AETrainer:

    def __init__(self, cfg: AETrainingConfig):
        self.cfg = cfg
        self.callbacks = cfg.callbacks or []

    def _call_callbacks(self, hook, state):
        for cb in self.callbacks:
            getattr(cb, hook, lambda x: None)(state)

    def train(self, model, X_train, X_val=None):
        device = self.cfg.device
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.lr)
        criterion = nn.MSELoss()

        # ---- DataLoaders ----
        train_loader = DataLoader(
            AEDataset(X_train),
            batch_size=self.cfg.batch_size,
            shuffle=self.cfg.shuffle,
            num_workers=self.cfg.num_workers
        )

        val_loader = None
        if X_val is not None:
            val_loader = DataLoader(
                AEDataset(X_val),
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
