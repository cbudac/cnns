"""
Test simulating components of DenseNet-121
"""

import torch
from cnn_backbones.densenet.densenet import DenseLayer, DenseBlock, Transition, DenseNet


def test_dense_layer():
    dl = DenseLayer(growth_rate=32, bn_size=4)
    # batch size 1, number of features from 1 conv layer: 64, size of width/height of img after conv: 56x56
    x = torch.randn(1, 64, 56, 56)
    y = dl(x)
    assert y.shape == (1, 32, 56, 56)


def test_dense_block():
    db = DenseBlock(bn_size=4, growth_rate=32, num_dense_layers=6)
    x = torch.randn(1, 64, 56, 56)
    y = db(x)
    # 256 = 64 (input size) + 6 (layers) * 32 (growth rate bottleneck)
    assert y.shape == (1, 256, 56, 56 )


def test_transition():
    t = Transition(out_channels=128)
    x = torch.randn(1, 256, 56, 56)
    y=t(x)
    assert y.shape == (1, 128, 28, 28)


def test_densenet():
    d = DenseNet(block_config=(6, 12, 24, 6))
    x = torch.randn(1,3, 224,224)
    y = d(x)