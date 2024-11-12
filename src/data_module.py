import os

import torch
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import FashionMNIST, Imagenette
from torchvision import transforms
import lightning as L


class BaseDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size: int = 32, shuffle: bool=True):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train = None
        self.val = None
        self.test = None
        self.predict = None


    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=self.batch_size)


class FashionMNISTDataModule(BaseDataModule):

    def prepare_data(self) -> None:
        # download the dataset
        FashionMNIST(self.data_dir, train=True, download=True)
        FashionMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,)),
                                        transforms.Resize(size=(224, 224))])

        if stage == "fit":
            mnist_full = FashionMNIST(self.data_dir, train=True, transform=transform)
            self.train, self.val = random_split(
                mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test = FashionMNIST(self.data_dir, train=False, transform=transform)

        if stage == "predict":
            self.predict = FashionMNIST(self.data_dir, train=False, transform=transform)


class ImagenetteDataModule(BaseDataModule):

    def prepare_data(self) -> None:
        if not [d for d in os.listdir(self.data_dir) if d.startswith("imagenette2-320")]:
            Imagenette(root = self.data_dir, size="320px", download=True)

    def setup(self, stage: str):
        transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                        transforms.ToTensor()])

        self.train = Imagenette(root = self.data_dir, split='train', size='320px', transform=transform)
        self.val = Imagenette(root=self.data_dir, split='val', size='320px', transform=transform)




