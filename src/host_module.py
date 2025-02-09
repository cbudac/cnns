from abc import ABC
from typing import Optional, Any

import torch
from lightning import LightningModule
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from torch.nn import functional as F
from torchmetrics.classification import MulticlassAccuracy

from cnn_backbones.densenet.densenet import DenseNet
from cnn_backbones.vgg.vgg import VGG


class HostModule(LightningModule, ABC):
    def __init__(self,
                 arch: str,
                 optimizer: OptimizerCallable = None,
                 scheduler: Optional[LRSchedulerCallable] = None,
                 scheduler_config: Optional[dict] = None,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model = self.create_model(arch, **kwargs)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_config = scheduler_config

        self.outputs = []

        # the num_classes should be pickup from config/dataset
        self.accuracy = MulticlassAccuracy(num_classes=10)


    def forward(self, x) -> Any:
        logits = self.model(x)
        return logits

    # ------------------------------------------------------------------------------------------------------------------
    def shared_step(self, batch, stage) -> STEP_OUTPUT:
        x = batch[0]
        y = batch[1]

        y_hat = self.forward(x)

        accuracy = self.accuracy(y_hat, y)
        self.log(f'step_{stage}_acc', accuracy, logger=True, on_step=True, on_epoch=False)

        loss = F.cross_entropy(y_hat, y)
        self.outputs.append(loss)
        self.log(f'step_{stage}_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    def training_step(self, batch) -> STEP_OUTPUT:
        return self.shared_step(batch, 'train')

    def validation_step(self, batch) -> STEP_OUTPUT:
        return self.shared_step(batch, "val")

    # ------------------------------------------------------------------------------------------------------------------
    def shared_epoch_end(self, stage) -> None:
        loss = torch.stack(self.outputs).mean()
        self.log(f'epoch_{stage}_loss', loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)

        accuracy = self.accuracy.compute()
        self.log(f'epoch_{stage}_accuracy', accuracy, prog_bar=False, logger=True, on_epoch=True, on_step=False)

        self.outputs.clear()
        self.accuracy.reset()

    def on_train_epoch_end(self) -> None:
        self.shared_epoch_end(stage = "train")

    def on_validation_epoch_end(self) -> None:
        self.shared_epoch_end(stage = "val")

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

    def create_model(self, arch: str, **kwargs) -> torch.nn.Module:
        archs = [
            VGG,
            DenseNet
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