import numpy as np



class DataModule:

    def __init__(self, train_path, val_path, y_val_path):
        self.train_path = train_path
        self.val_path = val_path
        self.y_val_path = y_val_path

    def load(self):
        X_train = np.load(self.train_path)
        X_val = np.load(self.val_path)
        y_val = np.load(self.y_val_path)
        return X_train, X_val, y_val
