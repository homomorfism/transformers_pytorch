import os

import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


class FRUITSDataloader(pl.LightningDataModule):
    def __init__(self, _path: str, test_size=0.1, batch_size=4, num_workers=4, shuffle=True):
        super(FRUITSDataloader, self).__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.training_path = os.path.join(_path, "data/fruits-360/Training")
        assert os.path.isdir(self.training_path)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
        ])

        dataset = ImageFolder(root=self.training_path, transform=self.transform)

        train_size = int((1 - test_size) * len(dataset))
        val_size = len(dataset) - train_size
        print(f"FRUITS dataset: train size={train_size}, val size={val_size}")

        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def classes(self):
        return os.listdir(self.training_path)
