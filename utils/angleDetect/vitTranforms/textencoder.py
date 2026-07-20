import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from typing import Union, List
from vitTranforms.simple_tokenizer import SimpleTokenizer as _Tokenizer
from packaging import version

import os, sys
filePath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(filePath)

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

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

#-O 专门配合文本编码使用
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                #-O 注意看这里的FeedForward不是全连阶层，里面还有网络构建，并且mlp_dim是隐藏层的层数
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

#-O 文本编码器，先用在说，又没搞过NLP鬼知道他是怎么把词变成向量的。
class textEconder(nn.Module):
    def __init__(self,vocab_size, dim, depth, heads, dim_head, mlp_dim, dropout=0., context_length=1):
        super().__init__()
        #-O context_length代表一次性读取词的最大个数
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, dim)

        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, dim))
        nn.init.xavier_normal_(self.positional_embedding)

        self.ln_final = LayerNorm(dim)  
        #-O 为什么这里要把初始化参数设为0，难道不会影响后续参数的更新嘛
        # self.text_projection = nn.Parameter(torch.empty(dim, dim))

        self.text_projection = nn.Parameter(torch.randn(dim, dim))

        self.transformer =  Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self._tokenizer = _Tokenizer()

    def forward(self, text):

        x = self.token_embedding(text)

        #-O 问题原来在这里， 你他妈的， 唉
        #-O 这东西初始参数值要先正常初始化一下，不然不能用，只怪自己没弄过NLP
        x = x + self.positional_embedding

        x = x.permute(1, 0, 2)  # NLD -> LND
        #-O 问题出在这个transformer上面,导致编码错误
        x = self.transformer(x)
        # print("text_features:",x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        #-O 这段代码没怎么搞清楚，但目前看来影响应该不大。
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        
        x = x @ self.text_projection

        return x
    
    def tokenize(self, texts: Union[str, List[str]]) -> torch.IntTensor:
        if isinstance(texts, str):
            texts = [texts]

        all_tokens = [self._tokenizer.encode(text) for text in texts]
        result = torch.tensor(all_tokens, dtype=torch.int)  # [batch_size, 1]
        return result



       
if __name__ == "__main__":
    textencoder = textEconder(
                vocab_size = 102, 
                dim = 512, 
                depth = 6, 
                heads = 16, 
                dim_head = 32, 
                mlp_dim = 1024, 

    )
    token = textencoder.tokenize(["101"])
    endtext = textencoder(token)
    print(endtext.shape)