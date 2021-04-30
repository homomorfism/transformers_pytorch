import pytorch_lightning as pl
import torch.nn as nn
from matplotlib import pyplot as plt

from models.utils import FeedForwardNetwork, Attention


class TransformerModel(pl.LightningModule):
    def __init__(self,
                 num_encoders: int,
                 dim: int,
                 heads: int,
                 dim_head: int,
                 dropout: float,
                 mlp_head: int,
                 displ_attention=False):
        """

        :param dim: d_model
        """
        super(TransformerModel, self).__init__()

        encoder = []

        self.fig, self.ax = plt.subplots(ncols=heads, nrows=num_encoders)

        for ii in range(num_encoders):
            encoder += [
                nn.LayerNorm(normalized_shape=dim),
                Attention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout, displ_attention=displ_attention,
                          ax=self.ax[ii]),
                nn.LayerNorm(normalized_shape=dim),
                FeedForwardNetwork(in_features=dim, hidden_features=mlp_head, out_features=dim, dropout=dropout)
            ]

        self.model = nn.Sequential(*encoder)

    def forward(self, x):
        x = self.model(x)

        plt.show()

        return x
