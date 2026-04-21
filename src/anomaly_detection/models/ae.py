from dataclasses import dataclass
from anomaly_detection.models.base import AnomalyModel

import torch
import torch.nn as nn


# dataclass for Model
@dataclass
class AEConfig:
    input_dim : 11


# Model
class AE(nn.Module):

    def __init__(self, cfg: AEConfig):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(cfg.input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, cfg.input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))




# Trainer
class AETrainer():
	# for now very simpel, later, also rain with x_val
	def train(self, model, X_train, X_val): # pass model here, not need to save state
		optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
		criterion = nn.MSELoss()

		# ---- Train (only normal data) ----
		X_t = torch.tensor(X_train, dtype=torch.float32)

		for epoch in range(50): # pass it later to training config!!!!!
		    optimizer.zero_grad()
		    recon = model(X_t)
		    loss = criterion(recon, X_t)
		    loss.backward()
		    optimizer.step()

		return model



# wrapper
class AutoencoderModel(AnomalyModel):
    
    def __init__(self, model, trainer):
        self.model = model
        self.trainer = trainer # injected, Only NNs see the Trainer


    def fit(self, X_train_prep, X_val_prep=None):
        self.trainer.train(self.model, X_train_prep, X_val_prep)


    def get_scores(self, X):
	    X_t = torch.tensor(X, dtype=torch.float32)
	    recon = self.model(X_t).detach()
	    error = ((X_t - recon) ** 2).mean(dim=1).numpy()
	    return error