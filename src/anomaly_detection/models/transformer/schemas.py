from dataclasses import dataclass


# ============================================================
# Configuration
# ============================================================

@dataclass
class TransformerAEConfig:
    input_dim: int
    seq_len: int
    d_model: int = 64
    nhead: int = 4
    num_encoder_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1
