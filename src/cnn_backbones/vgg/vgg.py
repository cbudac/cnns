from collections import OrderedDict
from enum import StrEnum

import torch.nn as nn
from torchinfo import summary
from logging_utils import get_logger

logger = get_logger(__name__)

class VGGConfig(StrEnum):
    A = "A"


class ConvLayer(nn.Sequential):
    def __init__(self, out_channels: int, use_batch_norm: bool = True):
        all_modules = [
            nn.LazyConv2d(out_channels=out_channels, kernel_size=(3, 3), stride=1, padding=1),
            nn.LazyBatchNorm2d(),
            nn.ReLU()
        ]
        # remove batch norm if not required - batch norm is not part of the VGG paper
        used_modules = filter(lambda module: (not isinstance(module, nn.LazyBatchNorm2d) or use_batch_norm), all_modules)
        super().__init__(*used_modules)


class Block(nn.Sequential):
    def __init__(self, conv_layers: int, out_channels: int, use_batch_norm: bool = True):
        super().__init__()

        for i in range(conv_layers):
            self.append(ConvLayer(out_channels=out_channels, use_batch_norm=use_batch_norm))

        self.append(nn.MaxPool2d(kernel_size=(2, 2), stride=2))


class VGG(nn.Module):
    """
    Without BatchNorm I could not overfit on MnistFashion
    """
    def __init__(self,
                 config: VGGConfig,
                 use_batch_norm: bool = True,
                 use_dropout: bool = True):
        super().__init__()
        logger.info(f"VGG config: {config}, use_batch_norm: {use_batch_norm}, use_dropout: {use_dropout}")
        blocks_definition = VGG.get_blocks_definition(config)
        self.blocks = nn.Sequential(OrderedDict([(f'block{idx}',
                                                  Block(*block, use_batch_norm=use_batch_norm)) for idx, block in enumerate(blocks_definition)]))

        fcl_modules = [nn.Flatten(),
                       nn.LazyLinear(out_features=4096),
                       nn.LazyBatchNorm1d(),
                       nn.ReLU(),
                       nn.Dropout(0.2),
                       nn.Linear(in_features=4096, out_features=10),
                       nn.BatchNorm1d(num_features=10),
                       nn.ReLU(),
                       nn.Dropout(0.2)]

        fcl_modules = filter(lambda module: (not isinstance(module, nn.Dropout) or use_dropout) and
                                            (not (isinstance(module, nn.BatchNorm1d) or
                                                  isinstance(module, nn.LazyBatchNorm1d))  or use_batch_norm) , fcl_modules)

        self.fc_modules = nn.Sequential(*fcl_modules)

    def forward(self, x):
        y_hat = self.fc_modules(self.blocks(x))
        return y_hat

    @staticmethod
    def get_blocks_definition(config: VGGConfig):
        # standard configuration from VGG paper
        if config == VGGConfig.A:
            return ((1, 64),
                    (1, 128),
                    (2, 256),
                    (2, 512),
                    (2, 512))
        else:
            raise ValueError(f"Unknown VGG config: {config}")

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    model = VGG(VGGConfig.A,
                use_dropout=False,
                use_batch_norm=False)
    summary(model, input_size=(1, 1, 224, 224), device='cpu', depth=2, col_names=["input_size", "output_size", "num_params"])