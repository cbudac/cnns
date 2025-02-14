import os

import torch
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import FashionMNIST, Imagenette, OxfordIIITPet
from torchvision import transforms
import lightning as L

from logging_utils import get_logger
from utils import Stage

logger = get_logger(__name__)

class BaseDataModule(L.LightningDataModule):
    def __init__(self,
                 data_dir: str = "./",
                 **kwargs):

        super().__init__()
        logger.info(f"Data dir: {data_dir}, kwargs: {kwargs}")

        self.data_dir = data_dir
        self.kwargs = kwargs

        self.train = None
        self.val = None
        self.test = None
        self.predict = None


    def train_dataloader(self):
        return DataLoader(self.train, **self.kwargs)

    def val_dataloader(self):
        return DataLoader(self.val, **self.kwargs)

    def test_dataloader(self):
        return DataLoader(self.test,  **self.kwargs)

    def predict_dataloader(self):
        return DataLoader(self.predict,  **self.kwargs)

# ----------------------------------------------------------------------------------------------------------------------
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

        if stage == Stage.FIT:
            mnist_full = FashionMNIST(self.data_dir, train=True, transform=transform)
            self.train, self.val = random_split(
                mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
            )

        # Assign test dataset for use in dataloader(s)
        if stage == Stage.TEST:
            self.test = FashionMNIST(self.data_dir, train=False, transform=transform)

        if stage == Stage.PREDICT:
            self.predict = FashionMNIST(self.data_dir, train=False, transform=transform)

# ----------------------------------------------------------------------------------------------------------------------
class OxfordIIITPetDataModule(BaseDataModule):
    def prepare_data(self) -> None:
        # download dataset
        OxfordIIITPet(self.data_dir, download=True, target_types="segmentation")

    def setup(self, stage: str):
        transform = transforms.Compose([transforms.ToTensor()])
        
        if stage == Stage.FIT:
            ox_pet_full = OxfordIIITPet(self.data_dir, transform=transform)
            self.train, self.val = random_split(
                ox_pet_full, [0.9, 0.1], generator=torch.Generator().manual_seed(42)
            )


# ----------------------------------------------------------------------------------------------------------------------
class ImagenetteDataModule(BaseDataModule):

    def prepare_data(self) -> None:
        if not [d for d in os.listdir(self.data_dir) if d.startswith("imagenette2-320")]:
            Imagenette(root = self.data_dir, size="320px", download=True)

    def setup(self, stage: str):
        transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                        transforms.ToTensor()])

        self.train = Imagenette(root = self.data_dir, split='train', size='320px', transform=transform)
        self.val = Imagenette(root=self.data_dir, split='val', size='320px', transform=transform)



# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    dm = OxfordIIITPetDataModule(data_dir='../data')
    dm.prepare_data()