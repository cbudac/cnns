"""
This module contains metrics adaptors. The intention is to decouple the host from task specific metrics.
"""
from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
from torchmetrics.classification import MulticlassAccuracy

from utils import Stage, Loop


class BaseMetricsAdaptor(ABC):
    @abstractmethod
    def on_fit_start(self, device: torch.device) -> None:
        ...

    @abstractmethod
    def on_step(self, stage: str, y_hat, y) -> None:
        ...

    @abstractmethod
    def on_epoch_end(self, stage: str) -> None:
        ...


class MulticlassAccuracyAdaptor(BaseMetricsAdaptor):
    def __init__(self, log_func: Callable, num_classes: int):
        self.log_func = log_func
        self.train_accuracy = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.val_accuracy = MulticlassAccuracy(num_classes=num_classes, average="micro")

    def on_fit_start(self, device: torch.device) -> None:
        self.train_accuracy = self.train_accuracy.to(device)
        self.val_accuracy = self.val_accuracy.to(device)

    def on_step(self, loop: Loop, y_hat, y) -> None:
        if loop == Loop.TRAIN:
            acc_value =self.train_accuracy(y_hat, y)
            self.log_func(f'{loop}_acc_step', acc_value, logger=True, on_step=True, on_epoch=False)
        elif loop == Loop.VAL:
            acc_value = self.val_accuracy(y_hat, y)
            self.log_func(f'{loop}_acc_step', acc_value, logger=True, on_step=True, on_epoch=False)

    def on_epoch_end(self, loop: Loop) -> None:
        if loop == Loop.TRAIN:
            acc_value = self.train_accuracy.compute()
            self.log_func(f'{loop}_acc_epoch', acc_value)
            self.train_accuracy.reset()
        elif loop == Loop.VAL:
            acc_value = self.val_accuracy.compute()
            self.log_func(f'{loop}_acc_epoch', acc_value)
            self.val_accuracy.reset()