import pytorch_lightning as pl
import seaborn as sns
import torch.nn as nn
from einops import rearrange
from torch import einsum


class Attention(nn.Module):
    """
    Reference: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0., displ_attention=False, ax=None):
        super().__init__()
        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.displ_attention = displ_attention
        self.ax = ax

    def forward(self, x):
        residual = x
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        if self.displ_attention:
            assert self.ax is not None
            attention = attn.detach().squeeze().numpy()

            for ii in range(self.heads):
                sns.heatmap(attention[ii], ax=self.ax[ii], cbar=False)
                self.ax[ii].grid(False)
                self.ax[ii].set_xticks([])
                self.ax[ii].set_yticks([])
                self.ax[ii].axis('scaled')

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out) + residual


class FeedForwardNetwork(pl.LightningModule):

    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 out_features: int,
                 dropout: float):
        super(FeedForwardNetwork, self).__init__()
        assert in_features == out_features

        self.dense1 = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.gelu = nn.GELU()
        self.dense2 = nn.Linear(in_features=hidden_features, out_features=out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.dense1(x)
        x = self.gelu(x)
        x = self.dense2(x)
        x = self.dropout(x)

        return x + residual
