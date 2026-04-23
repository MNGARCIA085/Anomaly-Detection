import torch.nn as nn
from anomaly_detection.models.ae.schemas import AEConfig



# Model
class AE(nn.Module):

    def __init__(self, cfg: AEConfig):
        super().__init__()

        # ----- Encoder -----
        encoder_layers = []
        in_dim = cfg.input_dim
        for dim in cfg.encoder_dims:
            encoder_layers.append(nn.Linear(in_dim, dim))
            encoder_layers.append(nn.ReLU())
            in_dim = dim
        encoder_layers = encoder_layers[:-1]  # remove last ReLU if you want

        self.encoder = nn.Sequential(*encoder_layers)

        # ----- Decoder -----
        decoder_layers = []
        in_dim = cfg.encoder_dims[-1]
        for dim in cfg.decoder_dims:
            decoder_layers.append(nn.Linear(in_dim, dim))
            decoder_layers.append(nn.ReLU())
            in_dim = dim

        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))


# Builder
def build_model(cfg, runtime_params):

    input_dim = runtime_params["input_dim"]

    ae_cfg = AEConfig(
        input_dim=input_dim,
        encoder_dims=cfg.encoder_dims,
        decoder_dims=cfg.encoder_dims[:-1][::-1] # maybe change this later
    )

    return AE(ae_cfg)
