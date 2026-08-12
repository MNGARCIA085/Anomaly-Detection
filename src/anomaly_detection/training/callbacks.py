# callbacks interface
class Callback:
    def on_train_start(self, state): pass
    def on_epoch_start(self, state): pass
    def on_epoch_end(self, state): pass
    def on_train_end(self, state): pass


class PrintLossCallback(Callback):
    def on_epoch_end(self, state):
        print(f"Epoch {state.epoch} - Train Loss: {state.train_loss:.4f} - Val Loss: {state.val_loss:.4f}")


class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.best = float("inf")
        self.counter = 0

    def on_epoch_end(self, state):
        if state.val_loss is None:
            return

        if state.val_loss < self.best:
            self.best = state.val_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            print('ES triggreed')
            state.stop_training = True



#---------------CREATE CALLBAKCS TO USE DYNAMICALLY WITH CONFIG----------#


CALLBACK_REGISTRY = {
    "print_loss": PrintLossCallback,
    "early_stopping": EarlyStopping,
}


def create_callback(name, **params):
    try:
        callback_cls = CALLBACK_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown callback: {name}. "
            f"Available: {list(CALLBACK_REGISTRY)}"
        )

    return callback_cls(**params)


def create_callbacks(cfg):
    callbacks = [] # I can put defaults here if i want
            #EarlyStopping(patience=3),
            #PrintLossCallback(),

    for callback_cfg in cfg:
        callbacks.append(
            create_callback(
                callback_cfg["name"],
                **callback_cfg.get("params", {}),
            )
        )

    return callbacks



def sample_callbacks(trial, cfg):
    callbacks = []

    # Enable/disable callbacks
    if cfg.print_loss:

        # if i want to sample it
        """
        print_loss = trial.suggest_categorical(
            "training.callbacks.print_loss",
            [True, False],
        )
        """

        print_loss = True
        if print_loss:
            callbacks.append({
                "name": "print_loss",
                "params": {},
            })

    if cfg.early_stopping:
        early_stopping = trial.suggest_categorical(
            "training.callbacks.early_stopping",
            [True, False],
        )

        if early_stopping:
            patience = trial.suggest_int(
                "training.callbacks.early_stopping.patience",
                cfg.patience.low,
                cfg.patience.high,
            )

            callbacks.append({
                "name": "early_stopping",
                "params": {
                    "patience": patience,
                },
            })

    return callbacks




