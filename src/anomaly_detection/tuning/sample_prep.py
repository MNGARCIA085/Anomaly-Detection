

#-----window size-------#
def sample_window_size(trial, cfg):
    """Sample window size for tuning."""
    window_size = trial.suggest_categorical(
        "data.windowing.size",
        cfg.choices,
    )

    return window_size



#---------scaler---------#
def sample_scaler(trial, cfg):
    """ Sample scaler for tuning"""
    scaler_name = trial.suggest_categorical(
        "prep.scaler.name",
        cfg.names,
    )

    return {
        "name": scaler_name,
        "params": {},
    }


#------------reducer---------#
def sample_dimensionality_reducer(trial, cfg):

    if not cfg.enabled:
        return {
            "enabled": False,
            "name": None,
            "params": {},
        }

    name = trial.suggest_categorical(
        "prep.dimensionality.name",
        cfg.names,
    )

    params = {}

    if name == "pca":
        params["n_components"] = trial.suggest_int(
            "prep.dimensionality.n_components",
            cfg.n_components.low,
            cfg.n_components.high,
        )

    return {
        "enabled": True,
        "name": name,
        "params": params,
    }



#----------feature selection--------------#
def sample_feature_selection(trial, cfg):

    if not cfg.enabled:
        return {
            "enabled": False,
        }

    name = trial.suggest_categorical(
        "prep.feature_selection.name",
        cfg.names,
    )

    threshold = trial.suggest_float(
        "prep.feature_selection.threshold",
        cfg.threshold.low,
        cfg.threshold.high,
    )

    return {
        "enabled": True,
        "name": name,
        "params": {
            "threshold": threshold,
        },
    }


#---------imputation-------------#
def sample_imputation(trial, cfg):
    """Sample imputation for tuning."""

    enabled = trial.suggest_categorical(
        "prep.imputation.enabled",
        [True, False],
    )

    if not enabled:
        return {
            "enabled": False,
        }

    name = trial.suggest_categorical(
        "prep.imputation.name",
        cfg.names,
    )

    strategy = trial.suggest_categorical(
        "prep.imputation.strategy",
        cfg.strategy.choices,
    )

    return {
        "enabled": True,
        "name": name,
        "params": {
            "strategy": strategy,
        },
    }



#---------transforms-------------#
def sample_transform(trial, cfg):

    # sample -> sometimes enabled sometimes not
    enabled = trial.suggest_categorical(
        "prep.transform.enabled",
        [True, False],
    )

    if not enabled:
        return {"enabled": False}

    name = trial.suggest_categorical(
        "prep.transform.name",
        cfg.names,
    )

    return {
        "enabled": True,
        "name": name,
    }