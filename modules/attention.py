from torch import nn


#输出层由于维度变化，没有使用残差连接######################
class AttnBlock(nn.Module):
    def __init__(self,in_d,oud_d,num_heads,dropout_prob=0.2):
        super().__init__()

        self.q=nn.Linear(in_d,in_d,bias=False)
        self.k = nn.Linear(in_d, in_d, bias=False)
        self.v = nn.Linear(in_d, in_d, bias=False)
        self.attn=nn.MultiheadAttention(in_d,num_heads=num_heads,dropout=dropout_prob)

        self.layernorm=nn.LayerNorm(in_d)

        self.fc=nn.Linear(in_d,oud_d)

        self.activation=nn.ReLU()

        self.drop_out=nn.Dropout(dropout_prob)

        self.layernorm_out=nn.LayerNorm(oud_d)

    def forward(self,x):

        q = self.activation(self.q(x))
        k = self.activation(self.k(x))
        v = self.activation(self.v(x))

        x1,_=self.attn(q,k,v)
        x=self.layernorm(x1+x)

        x=self.drop_out(self.activation(self.fc(x)))

        x=self.layernorm_out(x)

        return x