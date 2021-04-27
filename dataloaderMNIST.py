import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


class MNISTDataLoader(pl.LightningDataModule):
    def __init__(self, batch_size: int, name='mnist'):
        super(MNISTDataLoader, self).__init__()
        assert name == 'mnist' or name == 'fruit'

        self.batch_size = batch_size
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
        ])

        if name == 'mnist':
            self.train_dataset = MNIST(root='data/', train=True, transform=self.transforms, download=True)
            self.val_dataset = MNIST(root='data/', train=False, transform=self.transforms, download=True)

        else:
            raise NotImplemented()

        print(f"shape of train dataset[0]: {self.train_dataset[0]}")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )
