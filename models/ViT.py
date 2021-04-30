import einops
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from models.TransformerModel import TransformerModel
from models.preprocess import ImageToEmbedding, PositionalEncoding


class ViT(pl.LightningModule):

    def __init__(self,
                 image_size: int,
                 patch_size: int,
                 num_channels: int,
                 num_classes: int,
                 d_model: int,
                 num_blocks: int,
                 num_heads: int,
                 mvp_head: int,
                 dropout: float,
                 displ_attention=False
                 ):
        """
        Transformer model
        :param image_size: input image size
        :param patch_size: patches images size
        :param num_classes: num classes to classify
        :param d_model: length of encoder representation
        :param num_blocks: number of encoder/decoders blocks
        :param num_heads: numbers of parallel trainings in attention
        :param mvp_head: classificator hidden layer length
        :param dropout: 0.1
        """
        super(ViT, self).__init__()

        self.image_to_patch = ImageToEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            n_channels=num_channels,
            d_model=d_model
        )

        self.class_embedding = nn.Parameter(
            torch.randn(1, 1, d_model)
        )

        # n_position = num_patches + 1
        self.pos_encoding = PositionalEncoding(d_model=d_model, num_patches=(image_size // patch_size) ** 2)
        self.dropout = nn.Dropout(dropout)

        self.transformer = TransformerModel(
            num_encoders=num_blocks,
            dim=d_model,
            dim_head=64,
            mlp_head=mvp_head,
            heads=num_heads,
            dropout=dropout,
            displ_attention=displ_attention,
        )

        self.mvp = nn.Linear(in_features=d_model, out_features=num_classes)

        self.loss = nn.CrossEntropyLoss()

        # self.val_accuracy = pl.metrics.Accuracy()
        self.val_f1 = pl.metrics.F1(num_classes=num_classes)

    def forward(self, image):
        batch_size = image.size(0)
        patch_embeddings = self.image_to_patch(image)  # x.shape={batch_size, num_patches, d_model}

        class_embeddings = einops.repeat(self.class_embedding,
                                         '() 1 d_model -> batch_size 1 d_model', batch_size=batch_size)

        patch_embeddings = torch.cat((class_embeddings, patch_embeddings), dim=1)
        patch_embeddings = self.pos_encoding(patch_embeddings)
        patch_embeddings = self.dropout(patch_embeddings)

        output = self.transformer(patch_embeddings)
        output = self.mvp(output[:, 0])

        return output

    def configure_optimizers(self):
        optim = Adam(self.parameters(), lr=0.0005)
        sched = StepLR(optimizer=optim, step_size=5, gamma=0.5, )

        return [optim, ], [sched, ]

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)

        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        _, pred = torch.max(logits, dim=1)
        loss = self.loss(logits, y)

        self.log('val_loss', loss)
        self.log('val_f1', self.val_f1(pred, y))

        return loss

    def validation_epoch_end(self, outputs):
        mean_loss = sum(outputs) / len(outputs)
        print(f"Val loss: {mean_loss}, f1: {self.val_f1.compute()}")
