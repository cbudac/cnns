### Overview
This repo contains my implementation of several cnn backbones.
So far, I have implemented the following backbones:
- VGG
- DenseNet

The implementation is done in Pytorch, using PytorchLightning CLI and Tensorboard for logging.

### How to fit the model:
```
python src/cli.py fit  -c run_configs/fit_densenet_config.yaml
python src/cli.py fit  -c run_configs/fit_vgg_config.yaml
```

### Todo:
- enhance the data modules to allow control from config file for number of workers etc.
- add torch metrics and calculate classification specific metrics