import torch

import torch
from torch.utils.data import random_split
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms


def test_oxford_pet_dataset():
    data_dir='./data'
    target_transform =  transforms.Compose([transforms.PILToTensor(), 
                                            transforms.Lambda(lambda x: torch.squeeze(x)),
                                            transforms.Lambda(lambda x: x.to(torch.long))
                                            ])
    
    ox_pet_full = OxfordIIITPet(data_dir,  
                                transform=transforms.ToTensor(), 
                                target_transform=target_transform, 
                                target_types="segmentation")
    train, val = random_split(ox_pet_full, [0.9, 0.1], generator=torch.Generator().manual_seed(42))

    first_sample = train[0]

    assert first_sample[0].shape[0] == 3
    assert len(first_sample[1].shape) == 2


    assert torch.all(torch.eq(torch.unique(first_sample[1]), 
                              torch.tensor([1, 2, 3], dtype=torch.int64)))
