import numpy as np

from pathlib import Path

from anomaly_detection.inference.loader import load_from_config

import json


def main():


    with open("mock_model_store/models/model_001/config.json", "r") as f:
        config = json.load(f)


    model_dir = Path("mock_model_store/models/model_001")

    runner = load_from_config(
        config=config,
        model_dir=model_dir,
    )


    X = np.random.randn(20, 11)

    predictions = runner.predict(X)

    print("Input shape:", X.shape)
    print("Predictions:")
    print(predictions)


if __name__ == "__main__":
    main()
















