import math

import torch
from torch import nn
from modules.attention import AttnBlock
from modules.tokenizer import Tokenizer


def get_positional_encoding(d_model: int, max_len: int = 5000):
    encodings = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    two_i = torch.arange(0, d_model, 2, dtype=torch.float32)
    div_term = torch.exp(two_i * -(math.log(10000.0) / d_model))
    encodings[:, 0::2] = torch.sin(position * div_term)
    encodings[:, 1::2] = torch.cos(position * div_term)
    encodings = encodings.unsqueeze(1).requires_grad_(False)
    return encodings


class TextEncoder(nn.Module):
    def __init__(self,tokenizer:Tokenizer,embed_dim=256,max_len=2048,dim=[1,1,2,2,4]):

        super().__init__()
        self.embed_dim=embed_dim
        self.embedding=nn.Embedding(len(tokenizer),embedding_dim=embed_dim)
        self.tokenizer=tokenizer

        # 位置编码
        # self.positional_encodings = nn.Parameter(torch.zeros(max_len, 1, embed_dim), requires_grad=True)
        self.register_buffer('positional_encodings', get_positional_encoding(embed_dim, max_len))

        block=nn.ModuleList()
        self.num_blocks=len(dim)
        in_dim=(1,)+tuple(dim)

        for i in range(self.num_blocks):
            block_in=embed_dim*in_dim[i]
            block_out=embed_dim*dim[i]
            block.append(AttnBlock(block_in, block_out, num_heads=8, dropout_prob=0.2))

        self.block=block

    def forward(self,x):
        x=self.embedding(x)
        x=x.permute(1,0,2)
        SEP_token=self.embedding.weight[self.tokenizer.dictionary.token2id["<SEP>"]].unsqueeze(0).repeat(x.shape[1],1).unsqueeze(0)
        x=torch.cat([x,SEP_token],dim=0)
        pe=self.positional_encodings[:x.shape[0]].detach().requires_grad_(False)

        x=x*math.sqrt(self.embed_dim)+pe

        for i in range(self.num_blocks):
            x=self.block[i](x)

        return x

if __name__ == '__main__':
    tokenizer=Tokenizer(dictionary_path="../models/dictionary.gensim")
    encoder=TextEncoder(tokenizer=tokenizer,embed_dim=512)

    rand=torch.LongTensor([[1,5,7,2,6],[2,3,4,5,7]]).reshape(2,-1)

    out=encoder(rand)

    print(out.shape)