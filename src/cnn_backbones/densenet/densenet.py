import math
from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import MaxPool2d
from torchinfo import summary


class Conv7x7Layer(nn.Sequential):
    """
    This is the initial conv layer in the DenseNet paper
    """
    def __init__(self, growth_rate: int) -> None:
        super().__init__(nn.LazyConv2d(out_channels=2 * growth_rate, kernel_size=7, stride=2, padding=3),
                         nn.BatchNorm2d(num_features=2 * growth_rate),
                         nn.ReLU(inplace=True))


class Conv1x1Layer(nn.Sequential):
    """
    This is 1 x 1 conv layer in the DenseNet paper - the bottleneck layer
     - the bottleneck layer - reduces the number of input feature maps to improve computational efficiency (reduction to 4*k in paper)
    """
    def __init__(self, growth_rate: int, bn_size: int) -> None:
        super().__init__(nn.LazyBatchNorm2d(),
                         nn.ReLU(),
                         nn.LazyConv2d(out_channels=bn_size * growth_rate, kernel_size=1, stride=1, bias=False))


class Conv3x3Layer(nn.Sequential):
    """
    This is 3 x 3 conv layer in the DenseNet paper
    """
    def __init__(self, growth_rate: int, bn_size: int) -> None:
        super().__init__(nn.BatchNorm2d(bn_size * growth_rate),
                         nn.ReLU(),
                         nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))


class DenseLayer(nn.Sequential):
    """
    The DenseLayer contains a 1 x 1 conv bottleneck layer and the 3 x 3 conv layer
    The conv 1 x 1 layer forces the input into the conv 3 x 3 layer to be bn_size * growth_rate (4*k in the paper)
    The DenseLayer produces k (growth rate) feature maps
    """
    def __init__(self, growth_rate: int, bn_size: int) -> None:
        super().__init__(Conv1x1Layer(growth_rate=growth_rate, bn_size=bn_size),
                         Conv3x3Layer(growth_rate=growth_rate, bn_size=bn_size))


class DenseBlock(nn.ModuleDict):
    def __init__(self, num_dense_layers: int, bn_size: int, growth_rate: int) -> None:
        super().__init__()

        for i in range(num_dense_layers):
            layer = DenseLayer(growth_rate=growth_rate, bn_size=bn_size)
            self.add_module(f"denselayer_{i+1}", layer)

    def forward(self, in_features: Tensor) -> Tensor:
        out_features = [in_features]
        for name, layer in self.items():
            new_feature = layer(torch.cat(out_features, 1))
            out_features.append(new_feature)
        return torch.cat(out_features,1)


class Transition(nn.Sequential):
    def __init__(self, out_channels: int):
        super().__init__()
        self.norm = nn.LazyBatchNorm2d()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.LazyConv2d(out_channels=out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)


class DenseNet(nn.Sequential):
    """
    DenseNet + Bottleneck + Compression
    """
    def __init__(self,
                 block_config: Tuple[int, int, int, int] = (6,12, 24, 6),
                 growth_rate=32,
                 compression_factor = 0.5,
                 bn_size=4,
                 num_classes=10):
        super().__init__()

        # initial conv and pool layer
        self.append(Conv7x7Layer(growth_rate))
        self.append(MaxPool2d(kernel_size=3, stride=2, padding=1))


        # dense blocks and transitions
        num_features = 2 * growth_rate
        for i, num_layers in enumerate(block_config):
            dense_block = DenseBlock(num_dense_layers=num_layers, bn_size=bn_size, growth_rate=growth_rate)
            self.append(dense_block)

            num_features = num_features + num_layers * growth_rate
            num_output_features = math.floor(num_features * compression_factor)
            if i != len(block_config)-1:
                # add transition
                transition = Transition(out_channels=num_output_features)
                self.append(transition)
                num_features = num_output_features

        # final batch norm in the features backbone
        self.append(nn.BatchNorm2d(num_features=num_features))

        self.append(nn.ReLU(inplace=True))

        self.append(nn.AdaptiveAvgPool2d(output_size=(1,1)))
        self.append(nn.Flatten())

        # Linear layer
        self.append(nn.Linear(num_features, num_classes))

        # Official init from torch repo.
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.constant_(m.bias, 0)

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    model = DenseNet()
    summary(model, input_size=(1, 3, 224, 224), device='cuda', depth=3, col_names=["input_size", "output_size", "num_params"])