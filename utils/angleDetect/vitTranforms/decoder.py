import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import os, sys
filePath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(filePath)

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q , k, x):

        q = rearrange(q, 'b n (h d) -> b h n d',h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d',h=self.heads)
        v = rearrange(x, 'b n (h d) -> b h n d',h=self.heads) 

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        # print(self.to_out(out).shape)
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x, k, v):
        for attn, ff in self.layers:
            x = attn(x ,k, v) + x
            x = ff(x) + x

        return self.norm(x)

class crossViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

        #-O 进来之前先线性变化一下(V,Q的数值不能一样)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

    #-O 这里输入qkv
    def forward(self, x ,k ,v):
        #-O 这里不需要额外添加全连阶层，来进行线性变换
        x = self.q(x)
        #-O x.shape : torch.Size([1, 65, 1024])
        k = self.k(k)
        v = self.v(v)
        x = self.transformer(x,k,v)
        #-O x.shape :  torch.Size([16, 65, 1024])
        x = x[:, 0]
        #-O x.shape :  torch.Size([16, 1024])
        x = self.to_latent(x)

        return self.mlp_head(x)

if __name__ == "__main__":
    v = crossViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 5,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )

    total_params = sum(p.numel() for p in v.parameters())

    q = torch.randn(3, 65, 1024)
    k = torch.randn(3, 65, 1024)
    x = torch.randn(3, 65, 1024)

    preds = v(q, k, x)
    print(f"preds.shape:{preds.shape},total_params:{total_params}")
    # assert preds.shape == (3, 1000), 'correct logits outputted'