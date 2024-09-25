from typing import Any

import lightning.pytorch as pl
import torch
import torch.nn as nn
from elforecast.models import GRU, LSTM, ConvGRU, ConvLin, ConvResLike
from omegaconf.dictconfig import DictConfig
from torchmetrics.regression import MeanAbsolutePercentageError


model_registry = {
    "gru": GRU,
    "lstm": LSTM,
    "conv_lin": ConvLin,
    "conv_res_like": ConvResLike,
    "conv_gru": ConvGRU,
}


class Selector(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.loss_fn = nn.L1Loss()
        self.test_metric = MeanAbsolutePercentageError()

        self.cfg = cfg
        if cfg.model.name in model_registry:
            model_class = model_registry[cfg.model.name]
            self.model = model_class(cfg)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx=0):
        X_batch, y_batch = batch
        y_pred = self(X_batch)
        loss = self.loss_fn(y_pred, y_batch.squeeze(2))
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        X_batch, y_batch = batch
        y_pred = self(X_batch)
        loss = self.loss_fn(y_pred, y_batch.squeeze(2))
        metric = self.test_metric(y_pred.view(-1), y_batch.view(-1))
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_metric", metric, on_step=True, on_epoch=True, prog_bar=True)
        return {"val_loss": loss, "val_metric": metric}

    def predict_step(self, batch, batch_idx):
        return self(batch)

    def configure_optimizers(self) -> Any:
        opt = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.training.lr,
            weight_decay=self.cfg.training.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda epoch: self.cfg.training.lr_factor**epoch
        )
        return [opt], {
            "scheduler": scheduler,
            "interval": "step",
            "monitor": "train_loss",
        }
