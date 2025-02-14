from abc import ABC
from typing import Optional, Any

import torch
from lightning import LightningModule
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from torch.nn import functional as F

from cnn_backbones.densenet.densenet import DenseNet
from cnn_backbones.vgg.vgg import VGG
from metrics import BaseMetricsAdaptor
from sem_seg.unet.unet import UNet
from utils import Loop, create_instance


class HostModule(LightningModule, ABC):
    def __init__(self,
                 arch: str,
                 optimizer: OptimizerCallable = None,
                 scheduler: Optional[LRSchedulerCallable] = None,
                 scheduler_config: Optional[dict] = None,
                 metrics: Optional[list[dict]] = None,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model = self.create_model(arch, **kwargs)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_config = scheduler_config

        # check if metrics are configured and instantiate them
        self.metrics: list[BaseMetricsAdaptor] = []
        if metrics:
            for metric in metrics:
                metric = create_instance(metric["class_path"],self.log, **metric["init_args"])
                self.metrics.append(metric)

    def forward(self, x) -> Any:
        logits = self.model(x)
        return logits

    # ------------------------------------------------------------------------------------------------------------------
    def shared_step(self, batch, loop: Loop) -> STEP_OUTPUT:
        x = batch[0]
        y = batch[1]

        y_hat = self.forward(x)

        # invoke defined metrics
        for metric in self.metrics:
            metric.on_step(loop, y_hat, y)

        # compute loss
        loss = F.cross_entropy(y_hat, y)
        self.log(f'{loop}_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        return loss

    def training_step(self, batch) -> STEP_OUTPUT:
        return self.shared_step(batch, Loop.TRAIN)

    def validation_step(self, batch) -> STEP_OUTPUT:
        return self.shared_step(batch, Loop.VAL)

    # ------------------------------------------------------------------------------------------------------------------
    def shared_epoch_end(self, loop: Loop) -> None:
        for metric in self.metrics:
            metric: metric.on_epoch_end(loop)

    def on_train_epoch_end(self) -> None:
        self.shared_epoch_end(loop = Loop.TRAIN)

    def on_validation_epoch_end(self) -> None:
        self.shared_epoch_end(loop = Loop.VAL)

    # ------------------------------------------------------------------------------------------------------------------
    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = self.optimizer(self.parameters())

        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer)
            if self.scheduler_config is not None:
                return {
                            'optimizer': optimizer,
                            'lr_scheduler': {
                                    'scheduler':  scheduler,
                                    **self.scheduler_config
                                }
                            }
            else:
                return {
                            'optimizer': optimizer,
                            'lr_scheduler': scheduler
                        }
        else:
            return optimizer

    # ------------------------------------------------------------------------------------------------------------------
    def create_model(self, arch: str, **kwargs) -> torch.nn.Module:
        archs = [
            VGG,
            DenseNet,
            UNet
        ]
        archs_dict = {a.__name__.lower(): a for a in archs}
        try:
            model_class = archs_dict[arch.lower()]
        except KeyError:
            raise KeyError(
                "Wrong architecture type `{}`. Available options are: {}".format(
                    arch, list(archs_dict.keys())
                )
            )
        return model_class(**kwargs)

    def on_fit_start(self) -> None:
        # allow the metrics to gain insight about the device before fitting starts
        for metric in self.metrics:
            metric.on_fit_start(self.device)