import einops
import pytorch_lightning as pl
import torch
import torch.nn as nn


class PositionalEncoding(pl.LightningModule):

    def __init__(self, num_patches, d_model):
        super(PositionalEncoding, self).__init__()

        self.pos_table = nn.Parameter(torch.randn(1, num_patches + 1, d_model))

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)]


class ImageToEmbedding(pl.LightningModule):
    def __init__(self,
                 image_size: int,
                 patch_size: int,
                 n_channels: int,
                 d_model: int):
        super(ImageToEmbedding, self).__init__()

        assert image_size % patch_size == 0

        self.num_patches = (image_size // patch_size) ** 2

        self.patch_dim = n_channels * patch_size ** 2
        self.patch_size = patch_size

        self.dense = nn.Linear(in_features=self.patch_dim, out_features=d_model)

    def forward(self, image):
        image = einops.rearrange(
            image,
            'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
            p1=self.patch_size,
            p2=self.patch_size
        )

        image = self.dense(image)

        return image
