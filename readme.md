### Overview
This repo contains my implementation of several cnn backbones.
So far, I have implemented the following backbones:
- VGG (A)
- DenseNet

Main libraries: 
- Pytorch
- PytorchLightning (lightning[pytorch-extra] - for yaml config) CLI
- Torchmetrics
- Pytest
- Tensorboard for logging.

### How to fit the model:
```
python src/cli.py fit  -c run_configs/fit_densenet_config.yaml
python src/cli.py fit  -c run_configs/fit_vgg_config.yaml
```

### Todo:
- enhance the data modules to allow control from config file for number of workers etc.

### Observation
- for VGG, I had to add BatchNorm to have it overfit one batch and have the network behaving correctly