put buidl_ae_config like in main script

models/
  ae/
    model.py        ← architecture (nn.Module, etc.)
    trainer.py      ← training logic
    builder.py      ← AEBuilder (composition logic)
    config.py       ← build_ae_tuning_config, configs