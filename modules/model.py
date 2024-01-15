import math

import torch
from torch import nn
from modules.attention import AttnBlock

class VidEmbeddingWithPosition(nn.Module):
    def __init__(self,embed_dim=256,max_len=2048):
        super(VidEmbeddingWithPosition, self).__init__()
        self.embed_dim=embed_dim
        # 位置编码
        self.positional_encodings = nn.Parameter(torch.zeros(max_len, 1, embed_dim), requires_grad=True)

    def forward(self,vid_token,single_token_length):
        token_num=vid_token.shape[0]//single_token_length
        offset=vid_token.shape[0] % single_token_length
        pe=self.positional_encodings[:token_num]
        pe = pe.repeat(1,1,single_token_length).view(vid_token.shape[0]-offset,1,self.embed_dim)

        if offset!=0:
            pe2=self.positional_encodings[token_num].unsqueeze(0).repeat(offset,1,1)
            pe=torch.cat([pe,pe2],dim=0)

        vid_token=vid_token*math.sqrt(self.embed_dim)+pe
        return vid_token

class VideoModel(nn.Module):
    def __init__(self, in_c=3,out_c=3,embed_dim=256, max_len=2048, enc_dim=[1, 2, 2],dec_dim=[2,2,1]):
        super(VideoModel, self).__init__()

        self.vid_embedding=VidEmbeddingWithPosition(max_len=max_len,embed_dim=embed_dim)
        self.embed_dim=embed_dim
        enc_block=nn.ModuleList()
        self.enc_blocks_num=len(enc_dim)
        in_dim=(1,)+tuple(enc_dim)

        for i in range(self.enc_blocks_num):
            block_in=embed_dim*in_dim[i]
            block_out= embed_dim * enc_dim[i]
            enc_block.append(AttnBlock(block_in, block_out, num_heads=8, dropout_prob=0.2))

        self.enc_block=enc_block

        self.concat_atten1=nn.MultiheadAttention(embed_dim*dec_dim[0],num_heads=16)
        self.concat_atten2 = nn.MultiheadAttention(embed_dim * dec_dim[0], num_heads=16)

        dec_block=nn.ModuleList()
        self.dec_blocks_num=len(dec_dim)
        in_dim=(dec_dim[0],)+tuple(dec_dim)

        self.layernorm=nn.LayerNorm(embed_dim*dec_dim[0])
        self.activation=nn.ReLU()
        self.fc=nn.Linear(embed_dim*dec_dim[0],embed_dim*dec_dim[0])

        for i in range(self.dec_blocks_num):
            block_in=embed_dim*in_dim[i]
            block_out=embed_dim*dec_dim[i]
            dec_block.append(AttnBlock(block_in,block_out,num_heads=8,dropout_prob=0.2))

        self.dec_block=dec_block

    def encode(self,text_token,vid_token,single_token_length):
        vid_token=self.vid_embedding(vid_token,single_token_length)
        for i in range(self.enc_blocks_num):
            vid_token=self.enc_block[i](vid_token)

        x=torch.cat([text_token,vid_token],dim=0)
        x1,_=self.concat_atten1(x,x,x)
        x2, _ = self.concat_atten2(x1, x1, x1)

        x3=self.layernorm(x1+x2)
        x4=self.activation(self.fc(x3))

        x=self.layernorm(x3+x4)

        return x

    def decode(self,token):

        for i in range(self.dec_blocks_num):
            token=self.dec_block[i](token)

        return token

class MLP(nn.Module):
    def __init__(self,embed_dim=256,num_classes=2,dropou_prob=0.2):
        super(MLP, self).__init__()

        self.atten1 = AttnBlock(embed_dim, embed_dim, num_heads=8, dropout_prob=0.2)

        self.fc1=nn.Linear(embed_dim,embed_dim//2)

        self.fc2=nn.Linear(embed_dim//2,embed_dim//4)

        self.fc3=nn.Linear(embed_dim//4,num_classes)

        self.drop_out=nn.Dropout(dropou_prob)

        self.activation=nn.ReLU()

        self.layernorm=nn.LayerNorm(embed_dim)

        self.softmax=nn.Softmax(dim=1)

    def forward(self,x):

        x1=self.atten1(x)
        x1=self.layernorm(x+x1)
        x=x1[-1]
        x = self.drop_out(self.activation(self.fc1(x)))
        x = self.drop_out(self.activation(self.fc2(x)))
        x = self.drop_out(self.activation(self.fc3(x)))

        return  self.softmax(x)



if __name__ == '__main__':

    model=VidEmbeddingWithPosition(embed_dim=3)

    res=model(torch.LongTensor([[[2,3,4]],[[1,10,8]],[[5,6,3]],[[87,43,23]]]),single_token_length=2)
    print(res)