from collections import OrderedDict
from typing import List, Tuple

import torch.nn as nn
from torchinfo import summary


class ConvLayer(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )

class Block(nn.Sequential):
    def __init__(self, conv_layers: int, in_channels: int, out_channels: int):
        super().__init__()

        for i in range(conv_layers):
            layer_in_channels = in_channels if i == 0 else out_channels
            self.append(ConvLayer(in_channels=layer_in_channels, out_channels=out_channels))

        self.append(nn.MaxPool2d(kernel_size=(2, 2), stride=2))


class VGG(nn.Module):
    """
    Without BatchNorm I could not overfit on MnistFashion
    """
    def __init__(self,
                 blocks: List[Tuple[int, int, int]] = ((2, 3, 64),
                                                       (2, 64, 128),
                                                       (4, 128, 256),
                                                       (4, 256, 512),
                                                       (4, 512, 512)),
                 use_batch_norm: bool = True,
                 use_dropout: bool = True):
        super().__init__()
        self.blocks = nn.Sequential(OrderedDict([(f'block{idx}', Block(*block)) for idx, block in enumerate(blocks)]))

        fcl_modules = [nn.Flatten(),
                       nn.LazyLinear(out_features=4096),
                       nn.BatchNorm1d(num_features=4096),
                       nn.ReLU(),
                       nn.Dropout(0.2),
                       nn.Linear(in_features=4096, out_features=10),
                       nn.BatchNorm1d(num_features=10),
                       nn.ReLU(),
                       nn.Dropout(0.2)]

        fcl_modules = filter(lambda module: (not isinstance(module, nn.Dropout) or use_dropout) and
                                            (not isinstance(module, nn.BatchNorm1d) or use_batch_norm) , fcl_modules)

        self.fc_modules = nn.Sequential(*fcl_modules)

    def forward(self, x):
        y_hat = self.fc_modules(self.blocks(x))
        return y_hat


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    model = VGG(blocks=[(1, 1, 16), (1, 16,  32), (2, 32, 64), (2, 64, 128), (2, 128, 128)],
                use_dropout=False,
                use_batch_norm=False)
    summary(model, input_size=(1, 1, 224, 224), device='cpu')