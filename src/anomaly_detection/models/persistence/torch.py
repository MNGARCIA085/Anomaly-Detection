from pathlib import Path
import joblib
import torch



def save_torch_model(model, path):

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    torch.save(
        model.state_dict(),
        path / "weights.pt"
    )

    joblib.dump(
        model.config,
        path / "config.pkl"
    )

def load_torch_model(model_cls, path):

    path = Path(path)

    cfg = joblib.load(
        path / "config.pkl"
    )

    model = model_cls(cfg)

    model.load_state_dict(
        torch.load(
            path / "weights.pt",
            map_location="cpu"
        )
    )

    model.eval()

    return model