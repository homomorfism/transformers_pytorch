import os
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from dataloaders.dataloaderFRUITS360 import FRUITSDataloader
from dataloaders.dataloaderMNIST import MNISTDataLoader
from models.ViT import ViT


def main():
    root_path = ''

    parser = ArgumentParser()
    parser.add_argument('--dataset', help='dataset to train on: mnist/fruit', type=str, )
    dataset_name = parser.parse_args().dataset

    assert dataset_name in ['mnist', 'fruit']

    checkpoint_callback = ModelCheckpoint(
        os.path.join(root_path, 'checkpoints/'),
        filename=dataset_name,
        monitor='train_loss',
        save_last=False,
        save_top_k=1,
        mode='min'
    )

    # logger = TensorBoardLogger(save_dir=os.path.join(root_path, 'logs/'))

    trainer = pl.Trainer(
        gpus=0,
        callbacks=[checkpoint_callback],

        max_epochs=20,

    )

    if dataset_name == 'mnist':
        dataloader = MNISTDataLoader(_path=root_path, batch_size=128, num_workers=2)
        model = ViT(
            image_size=28,
            patch_size=7,
            num_channels=1,
            num_classes=10,
            d_model=128,
            num_blocks=6,
            num_heads=8,
            mvp_head=512,
            dropout=0.1,
        )

    else:
        dataloader = FRUITSDataloader(_path=root_path, batch_size=64, num_workers=2)
        num_classes = len(dataloader.classes())
        model = ViT(
            image_size=100,
            patch_size=10,
            num_channels=3,
            num_classes=num_classes,
            d_model=128,
            num_blocks=6,
            num_heads=8,
            mvp_head=512,
            dropout=0.1,
        )

    train_dataloader = dataloader.train_dataloader()
    val_dataloader = dataloader.val_dataloader()

    trainer.fit(model=model,
                train_dataloader=train_dataloader,
                val_dataloaders=val_dataloader)


if __name__ == '__main__':
    main()
