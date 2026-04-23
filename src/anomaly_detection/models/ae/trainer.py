import torch
import torch.nn as nn
from anomaly_detection.models.ae.schemas import AETrainingConfig

# Trainer
class AETrainer():

    def __init__(self, cfg: AETrainingConfig):
        self.lr = cfg.lr  
        self.epochs = cfg.epochs
        self.batch_size = cfg.batch_size

    # later -> add dataloaders...; callbacks


    # for now very simpel, later, also rain with x_val
    def train(self, model, X_train, X_val): # pass model here, not need to save state
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        # ---- Train (only normal data) ----
        X_t = torch.tensor(X_train, dtype=torch.float32)

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            recon = model(X_t)
            loss = criterion(recon, X_t)
            loss.backward()
            optimizer.step()

        return model