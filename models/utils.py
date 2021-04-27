import pytorch_lightning as pl
import torch.nn as nn
from einops import rearrange
from torch import einsum


class Attention(nn.Module):
    """
    Reference: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        residual = x
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')


        return self.to_out(out) + residual


class FeedForwardNetwork(pl.LightningModule):

    def __init__(self, in_features: int, hidden_features: int, out_features: int, dropout):
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