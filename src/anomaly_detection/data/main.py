

from .data import DataModule




# later centralize!!!!


from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[3]  # __file__ -> actual file location
# main.py → data → anomaly_detection → src → project root

TRAIN_PATH = BASE_DIR / "data" / "servers" / "X_part2.npy"
VAL_PATH = BASE_DIR / "data" / "servers" / "X_val_part2.npy"
Y_VAL_PATH = BASE_DIR / "data" / "servers" / "y_val_part2.npy"





if __name__=="__main__":
	data = DataModule(TRAIN_PATH, VAL_PATH, Y_VAL_PATH)

	X_train, X_val, y_val = data.load()

	print(X_train.shape, X_val.shape, y_val.shape)

	print(X_train[:3])

	print(y_val[:5])



#python -m anomaly_detection.data.main