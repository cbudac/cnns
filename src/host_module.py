from abc import ABC
from typing import Optional, Any

import torch
from lightning import LightningModule
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from torch.nn import functional as F

from densenet.densenet import DenseNet
from vgg.vgg import VGG


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

        self.test_step_outputs = []
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, x) -> Any:
        logits = self.model(x)
        return logits

    # ------------------------------------------------------------------------------------------------------------------
    def shared_step(self, batch, batch_idx, stage) -> STEP_OUTPUT:
        x = batch[0]
        y = batch[1]

        y_hat = self.forward(x)

        loss = F.cross_entropy(y_hat, y)
        self.log(f'step_{stage}_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        training_step_output = self.shared_step(batch, batch_idx, 'train')
        self.training_step_outputs.append(training_step_output)
        return training_step_output

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        validation_step_output = self.shared_step(batch, batch_idx, "val")
        self.validation_step_outputs.append(validation_step_output)
        return validation_step_output

    # ------------------------------------------------------------------------------------------------------------------
    def shared_epoch_end(self, outputs, stage, epoch) -> None:
        loss = torch.stack(outputs).mean()
        self.log(f'epoch_{stage}_loss', loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        outputs.clear()

    def on_train_epoch_end(self) -> None:
        self.shared_epoch_end(self.training_step_outputs, "train", self.current_epoch)

    def on_validation_epoch_end(self) -> None:
        self.shared_epoch_end(self.validation_step_outputs, "val", self.current_epoch)

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