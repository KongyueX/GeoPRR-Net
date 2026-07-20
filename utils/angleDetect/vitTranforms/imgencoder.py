import torch
from torch import nn

import os, sys
filePath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(filePath)

from vitTranforms.decoder import crossViT
from vitTranforms.encoder import ViT

class imgEncoder(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dropout = 0., emb_dropout = 0.):
        super().__init__()
        self.encoder = ViT(
                    image_size = image_size,
                    patch_size = patch_size,
                    num_classes = num_classes,
                    dim = dim,
                    depth = depth,
                    heads = heads,
                    mlp_dim = mlp_dim,
                    channels = channels,
                    dropout = dropout,
                    emb_dropout = emb_dropout
        )
    

        self.decoder = crossViT(
                    image_size = image_size,
                    patch_size = patch_size,
                    num_classes = num_classes,
                    dim = dim,
                    depth = depth,
                    heads = heads,
                    mlp_dim = mlp_dim,
                    channels = channels,
                    dropout = dropout,
                    emb_dropout = emb_dropout
        )

    def forward(self, img1, img2):
        # print(img1.shape)
        v = self.encoder(img1)
        k = v
        q = self.encoder(img2)
        # print(q.shape, k.shape, v.shape,)
        preds = self.decoder(q,k,v)

        return preds


if __name__ == "__main__":
    viTranforms = imgEncoder(
                  image_size = 256,
                  patch_size = 32,
                  num_classes = 512,
                  dim = 1024,
                  depth = 6,
                  heads = 16,
                  mlp_dim = 2048,
                  dropout = 0.1,
                  emb_dropout = 0.1

    )

    img1 = torch.randn(1, 3, 256, 256)
    img2 = torch.randn(1, 3, 256, 256)

    text = "100"

    preds = viTranforms(img1, img2)

    print(preds.shape)

    # assert preds.shape == (3, 5), 'correct logits outputted' 