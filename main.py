import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from dataloaderFRUITS360 import FRUITSDataloader
from dataloaderMNIST import MNISTDataLoader
from models.ViT import ViT

dataset_name = 'mnist'
assert dataset_name in ['fruit', 'mnist']


def main():
    checkpoint_callback = ModelCheckpoint(
        'checkpoints/',
        filename='mnist',
        monitor='train_loss',
        save_last=False,
        save_top_k=1,
        mode='min'
    )

    logger = TensorBoardLogger(save_dir='logs/')

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],

        # fast_dev_run=True,
        progress_bar_refresh_rate=50,
        logger=logger,
        max_epochs=1000

    )

    if dataset_name == 'mnist':
        dataloader = MNISTDataLoader(batch_size=4)
        model = ViT(
            image_size=28,
            patch_size=7,
            num_channels=1,
            num_classes=10,
        )

    else:
        dataloader = FRUITSDataloader(batch_size=4)
        num_classes = len(dataloader.classes())
        model = ViT(
            image_size=100,
            patch_size=10,
            num_channels=3,
            num_classes=num_classes,
        )

    train_dataloader = dataloader.train_dataloader()
    val_dataloader = dataloader.val_dataloader()

    trainer.fit(model=model,
                train_dataloader=train_dataloader,
                val_dataloaders=val_dataloader)


if __name__ == '__main__':
    main()
