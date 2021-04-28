import pytorch_lightning as pl
import torch.nn as nn

from models.utils import FeedForwardNetwork, Attention


class TransformerModel(pl.LightningModule):
    def __init__(self,
                 num_encoders: int,
                 dim: int,
                 heads: int,
                 dim_head: int,
                 dropout: float,
                 mlp_head: int):
        """

        :param dim: d_model
        """
        super(TransformerModel, self).__init__()

        encoder = []

        for _ in range(num_encoders):
            encoder += [
                nn.LayerNorm(normalized_shape=dim),
                Attention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout),
                nn.LayerNorm(normalized_shape=dim),
                FeedForwardNetwork(in_features=dim, hidden_features=mlp_head, out_features=dim, dropout=dropout)
            ]

        self.model = nn.Sequential(*encoder)

    def forward(self, x):
        x = self.model(x)

        return x
