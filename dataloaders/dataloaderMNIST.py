import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


class MNISTDataLoader(pl.LightningDataModule):
    def __init__(self, _path: str, batch_size: int, num_workers: int = 4):
        super(MNISTDataLoader, self).__init__()

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0, ], std=[1, ])
        ])

        self.train_dataset = MNIST(root=os.path.join(_path, 'data/'), train=True, transform=self.transforms,
                                   download=True)
        self.val_dataset = MNIST(root=os.path.join(_path, 'data/'), train=False, transform=self.transforms,
                                 download=True)

        print(f"shape of mnist train dataset[0][0]: {self.train_dataset[0][0].size()}")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
