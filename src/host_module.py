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

        # the num_classes should be pickup from config/dataset
        self.train_accuracy = MulticlassAccuracy(num_classes=10, average="micro")
        self.val_accuracy = MulticlassAccuracy(num_classes=10, average="micro")


    def forward(self, x) -> Any:
        logits = self.model(x)
        return logits

    # ------------------------------------------------------------------------------------------------------------------
    def shared_step(self, batch, stage) -> STEP_OUTPUT:
        x = batch[0]
        y = batch[1]

        y_hat = self.forward(x)

        if stage=="train":
            train_accuracy = self.train_accuracy(y_hat, y)
            self.log(f'{stage}_acc_step', train_accuracy, logger=True, on_step=True, on_epoch=False)
        elif stage=="val":
            val_accuracy = self.val_accuracy(y_hat, y)
            self.log(f'{stage}_acc_step', val_accuracy, logger=True, on_step=True, on_epoch=False)

        loss = F.cross_entropy(y_hat, y)
        self.log(f'{stage}_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        return loss

    def training_step(self, batch) -> STEP_OUTPUT:
        return self.shared_step(batch, 'train')

    def validation_step(self, batch) -> STEP_OUTPUT:
        return self.shared_step(batch, "val")

    # ------------------------------------------------------------------------------------------------------------------
    def shared_epoch_end(self, stage) -> None:
        if stage == 'train':
            train_accuracy = self.train_accuracy.compute()
            self.log(f'{stage}_acc_epoch', train_accuracy)
            self.train_accuracy.reset()
        elif stage == 'val':
            val_accuracy = self.val_accuracy.compute()
            self.log(f'{stage}_acc_epoch', val_accuracy)
            self.val_accuracy.reset()

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