from typing import Any

import lightning.pytorch as pl
import torch
import torch.nn as nn
from omegaconf.dictconfig import DictConfig
from torchmetrics.regression import R2Score


class ConvLin(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super(ConvLin, self).__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.loss_fn = nn.L1Loss()
        self.test_metric = R2Score()

        input_size = cfg.data.lseq
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=input_size,
                out_channels=self.cfg.model.params.conv.n_channels[0],
                kernel_size=self.cfg.model.params.conv.kernels[0],
                padding=self.cfg.model.params.conv.padding[0],
            ),
            nn.BatchNorm1d(self.cfg.model.params.conv.n_channels[0]),
            nn.ReLU(),
            nn.MaxPool1d(
                kernel_size=self.cfg.model.params.conv.maxpool[0],
                stride=self.cfg.model.params.conv.maxpool[1],
            ),
            nn.Dropout(self.cfg.model.params.conv.dropout),
            nn.Conv1d(
                in_channels=self.cfg.model.params.conv.n_channels[0],
                out_channels=self.cfg.model.params.conv.n_channels[1],
                kernel_size=self.cfg.model.params.conv.kernels[1],
                padding=self.cfg.model.params.conv.padding[1],
            ),
            nn.BatchNorm1d(self.cfg.model.params.conv.n_channels[1]),
            nn.ReLU(),
            nn.MaxPool1d(
                kernel_size=self.cfg.model.params.conv.maxpool[0],
                stride=self.cfg.model.params.conv.maxpool[1],
            ),
            nn.Dropout(self.cfg.model.params.conv.dropout),
            nn.Conv1d(
                in_channels=self.cfg.model.params.conv.n_channels[1],
                out_channels=self.cfg.model.params.conv.n_channels[2],
                kernel_size=self.cfg.model.params.conv.kernels[2],
                padding=self.cfg.model.params.conv.padding[2],
            ),
            nn.BatchNorm1d(self.cfg.model.params.conv.n_channels[2]),
            nn.ReLU(),
            nn.MaxPool1d(
                kernel_size=self.cfg.model.params.conv.maxpool[0],
                stride=self.cfg.model.params.conv.maxpool[1],
            ),
            nn.Dropout(self.cfg.model.params.conv.dropout),
            nn.Flatten(),
        )
        self.lin = nn.Sequential(
            nn.Linear(
                self.cfg.model.params.conv.n_channels[-1],
                self.cfg.model.params.lin.size[0],
            ),
            nn.BatchNorm1d(self.cfg.model.params.lin.size[0]),
            nn.ReLU(),
            nn.Dropout(self.cfg.model.params.lin.dropout),
            nn.Linear(
                self.cfg.model.params.lin.size[0], self.cfg.model.params.lin.size[1]
            ),
            nn.BatchNorm1d(self.cfg.model.params.lin.size[1]),
            nn.ReLU(),
            nn.Dropout(self.cfg.model.params.lin.dropout),
            nn.Linear(self.cfg.model.params.lin.size[1], input_size),
        )

    def forward(self, x):
        out = self.cnn(x)
        out = self.lin(out)
        return out

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx=0):
        X_batch, y_batch = batch
        y_pred = self(X_batch)
        loss = self.loss_fn(y_pred, y_batch)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        X_batch, y_batch = batch
        y_pred = self(X_batch)
        loss = self.loss_fn(y_pred, y_batch)
        metric = self.test_metric(y_pred.view(-1), y_batch.view(-1))
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_metric", metric, on_step=True, on_epoch=True, prog_bar=True)
        return {"val_loss": loss, "val_metric": metric}

    def predict_step(self, batch, batch_idx):
        return self(batch)

    def configure_optimizers(self) -> Any:
        opt = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg["training"]["lr"],
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
