import torch
import numpy as np
from torch import nn

import os, sys
filePath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(filePath)

from vitTranforms.imgencoder import imgEncoder
from vitTranforms.textencoder import textEconder


class meterClip(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dropout = 0., emb_dropout = 0.):
        super().__init__()
        self.imgencoder =   imgEncoder(
                            image_size = image_size,
                            patch_size = patch_size,
                            num_classes = num_classes,
                            dim = 1024,
                            depth = 6,
                            heads = 16,
                            mlp_dim = 2048,
                            channels = channels,
                            dropout = 0.1,
                            emb_dropout = 0.1

        )
    

        self.texteconder =  textEconder(
                            vocab_size = 102, 
                            dim = 512, 
                            depth = depth, 
                            heads = heads, 
                            dim_head = 32, 
                            mlp_dim = 1024, 

        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, img1, img2, text):
       
        #-O 张量尺寸对不上
        image_features = self.imgencoder(img1, img2)
        #-O 出现数字1-100的编码结果都一模一样，编码器有问题（已解决）
        #-O 这里直接把tokenize嵌入进来了
        # text = self.texteconder.tokenize(text)       
        text_features = self.texteconder(text)
        # print("text_features:",text_features)
        #-O torch.Size([1, 512])
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        #-O torch.Size([1, 512])
        #-O 问题在文本编码的身上，模型第一次运行结果里面都是nan
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        
        # print("image_features:",image_features)
        # print("text_features:",text_features)

        logit_scale = self.logit_scale.exp()
        #-O problem is here
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text

if __name__ == "__main__":
    meterclip = meterClip(
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

    
    img1 = torch.randn(4, 3, 256, 256)
    img2 = torch.randn(4, 3, 256, 256)
    text = torch.tensor([95, 34, 36, 74]).tolist()
    text = list(map(str, text))
    print(text)
    #-O
    # ['71', '9', '34', '49']
    text = meterclip.texteconder.tokenize(text)
    # text = ["0", "1", "11", "101"]
    
    meterclip.eval()
    torch.no_grad()
    
    predsImg, predsLabel = meterclip(img1, img2, text)

    print(predsImg)

    