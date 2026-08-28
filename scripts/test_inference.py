"""
model:
  type: ae

data:
  windowing:
    size: 1
"""

"""
from anomaly_detection.inference.loader import load_from_config


def main():

  config = {
    "model": {
      "type": "ae"
    },
    "data": {
        "windowing": {
            "size":1
        }
    }
  }


  runner = load_from_config(
    config=config,
    model_dir="mock_model_store/models/ae/v1",
  )

  print(runner)
  



if __name__=="__main__":
  main()

"""








from pathlib import Path

import numpy as np

from anomaly_detection.inference.loader import load_from_config


def main():

    config = {
        "model": {
            "type": "ae",
        },
        "data": {
            "windowing": {
                "size": 5,
            }
        },
    }

    model_dir = Path("mock_model_store/models/ae/v1")

    runner = load_from_config(
        config=config,
        model_dir=model_dir,
    )

    # Example input.
    # IMPORTANT: replace 11 with the number of features
    # expected by the saved preprocessor/model.
    X = np.random.randn(20, 11)

    predictions = runner.predict(X)

    print("Input shape:", X.shape)
    print("Predictions:")
    print(predictions)


if __name__ == "__main__":
    main()
















