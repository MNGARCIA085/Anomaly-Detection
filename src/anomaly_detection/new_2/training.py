

# =========================================================
# 2. CALLBACK
# =========================================================

class EarlyStopping:

    def __init__(self, patience=3, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta

        self.best = float("inf")
        self.counter = 0
        self.should_stop = False

    def __call__(self, val_loss):

        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True