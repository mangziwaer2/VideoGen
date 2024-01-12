import torch
from einops import rearrange
from torch import nn
from modules.text_encoder import TextEncoder
from modules.model import VideoModel,MLP
from modules.VQGAN.vqgan import VQModel
from modules.tokenizer import Tokenizer


class CompletionModel(nn.Module):
    def __init__(self,tokenizer:Tokenizer,embed_dim=256,max_len=2048):
        super(CompletionModel, self).__init__()
        self.embed_dim=embed_dim
        self.text_encoder=TextEncoder(tokenizer=tokenizer,embed_dim=embed_dim,max_len=max_len)
        self.vid_model=VideoModel(embed_dim=embed_dim,max_len=max_len)
        self.vqmodel=VQModel(n_embed=8192,embed_dim=embed_dim)
        self.next_state_model=MLP(embed_dim=embed_dim*4, num_classes=2)

        self.img_token_len=0
        self.img_token_shape=None

    def encode(self, text, img):
        print(img.shape)
        text_token=self.text_encoder(text)
        img_enc, qloss, _=self.vqmodel.encode(img)
        print(img_enc.shape)
        token_num=img_enc.shape[-1]*img_enc.shape[-2]
        self.img_token_shape=[img_enc.shape[-1],img_enc.shape[-2]]
        self.img_token_len=token_num
        z = rearrange(img_enc, 'b c h w -> b h w c').contiguous()
        vid_token= z.view(z.shape[1]*z.shape[2],z.shape[0], self.embed_dim)

        return text_token,vid_token,qloss

    def decode(self,text_token,vid_token,memory_length=15):
        SPAN_token = self.text_encoder.embedding.weight[self.text_encoder.tokenizer.dictionary.token2id["<SPAN>"]].unsqueeze(0).repeat(vid_token.shape[1],1).unsqueeze(0)
        vid_token_SPAN=torch.cat([vid_token,SPAN_token],dim=0)
        token=self.vid_model.encode(text_token,vid_token_SPAN,self.img_token_len)

        new_token=token[-self.img_token_len:]
        new_token=self.vid_model.decode(new_token)
        img_token=new_token.view(self.img_token_shape[0],self.img_token_shape[1],new_token.shape[1],self.embed_dim)#h w b c
        img_token=img_token.permute(2,3,0,1)#b c h w
        img=self.vqmodel.decode(img_token)
        new_vid_token=torch.cat([vid_token,new_token],dim=0)

        mf=self.next_state_model(token)

        new_vid_token=new_vid_token[-self.img_token_len*memory_length:]

        return img,new_vid_token,mf

if __name__ == '__main__':
    c_model=CompletionModel()

