import os

import hydra
import lightning.pytorch as pl
from elforecast.data import ElForecastDataModule
from elforecast.models import Selector
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    dm = ElForecastDataModule(cfg)
    model = Selector(cfg)

    logger = MLFlowLogger(
        experiment_name=cfg.model.name,
        tracking_uri=cfg.artifacts.mlflow_url,
    )

    callbacks = [
        pl.callbacks.DeviceStatsMonitor(),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.ModelCheckpoint(
            os.path.join(cfg.artifacts.checkpoint.dirpath, cfg.model.name),
            filename=cfg.model.name,
            save_top_k=cfg.artifacts.checkpoint.save_top_k,
            every_n_epochs=cfg.artifacts.checkpoint.every_n_epochs,
            save_last=cfg.artifacts.checkpoint.save_last,
        ),
    ]

    trainer = pl.Trainer(
        default_root_dir=cfg.artifacts.checkpoint.dirpath,
        max_epochs=cfg.training.n_epochs,
        deterministic=cfg["training"]["full_deterministic_mode"],
        log_every_n_steps=1,
        gradient_clip_val=cfg.training.clip_val,
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
