#Optuna/search-space sampling for training configuration


def sample_callbacks(trial, cfg):

    callbacks = []

    if cfg.print_loss:
        if trial.suggest_categorical(
            "training.callbacks.print_loss",
            [True, True], # True, False
        ):
            callbacks.append({
                "name": "print_loss",
                "params": {},
            })

    if cfg.early_stopping.enabled:
        enabled = trial.suggest_categorical(
            "training.callbacks.early_stopping.enabled",
            [True, True], # [True, False]
        )

        if enabled:
            patience = trial.suggest_int(
                "training.callbacks.early_stopping.patience",
                cfg.early_stopping.patience.low,
                cfg.early_stopping.patience.high,
            )

            callbacks.append({
                "name": "early_stopping",
                "params": {
                    "patience": patience,
                },
            })

    return callbacks