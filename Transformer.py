import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary


class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model%num_heads==0
        self.d_k = d_model//num_heads
        
        self.query = nn.Linear(d_model,d_model)
        self.key = nn.Linear(d_model,d_model)
        self.value = nn.Linear(d_model,d_model)

        self.outLinear = nn.Linear(d_model,d_model)
    def split_heads(self,x):
        b,seq_length,_ = x.size()
        x= x.view(b,seq_length,self.num_heads,self.d_k).permute(0,2,1,3)
        return x
    def forward(self,query,key,value,mask = None):
        query = self.split_heads(self.query(query))
        key = self.split_heads(self.key(key))
        value = self.split_heads(self.value(value))


        score = torch.matmul(query,key.transpose(-2,-1))/(self.d_k)**(1/2)
        if mask is not None:
            score =score.masked_fill(mask==0,-np.inf)
        Attention = F.softmax(score,dim=-1)
        out_Attention =  torch.matmul(Attention,value)

        b,_,s,_ = out_Attention.size()
        out_Attention = out_Attention.transpose(1,2).contiguous().view(b,s,self.d_model)

        out_Attention = self.outLinear(out_Attention)
        return out_Attention
class FeedForward(nn.Module):
    def __init__(self,d_model,d_MLP,dropout=None):
        super().__init__()
        self.fc1 = nn.Linear(d_model,d_MLP)
        self.fc2 = nn.Linear(d_MLP,d_model)
        self.dropoutlayer = nn.Sequential()
        if dropout is not None:
            self.dropoutlayer = nn.Sequential(nn.Dropout(dropout))
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.dropoutlayer(x)
        x = self.fc2(x)
        x = self.dropoutlayer(x)
        return x
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model,num_heads,d_MLP,dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.Attention = MultiHeadAttention(d_model,num_heads)
        self.FeedForward = FeedForward(d_model,d_MLP,dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x,mask =None):
             
        Attention_layer = self.Attention(x,x,x,mask)
        Attention_layer = self.dropout(Attention_layer)
        x = x+Attention_layer
        x =self.norm1(x)

        FeedForward_layer = self.FeedForward(x)
        x = x+FeedForward_layer
        x = self.norm2(x)
        return x
class TransformerEncoder(nn.Module):
    def __init__(self,d_model,num_heads,d_MLP,dropout,num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, num_heads, d_MLP,dropout) for _ in range(num_layers)]
        )
    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x
class PatchEmbedding(nn.Module):
    def __init__(self,in_shape:tuple,embedding_dim,patch_size : tuple ,dropout):
        super().__init__()
        # patch embedding
        # embedding_dim = in_channels*patch_size[0]*patch_size[1]
        in_channels,in_height,in_width = in_shape
        num_patch = in_height*in_width//(patch_size[0]*patch_size[1])
        self.patch_do = nn.Sequential(
            nn.Conv2d(in_channels,embedding_dim,kernel_size=patch_size,stride=patch_size,bias=False),
            nn.Flatten(2)
        )
        self.dropout = nn.Dropout(dropout)
        self.cls_token = nn.Parameter(torch.randn(size=(1,1,embedding_dim)),requires_grad=True)
        self.position_embedding = nn.Parameter(torch.randn(size = (1,1+num_patch,embedding_dim)),requires_grad=True)
    def forward(self,x):
        cls_token = self.cls_token.expand(x.size(0),-1,-1)
        x = self.patch_do(x).permute(0,2,1)
        x = torch.cat([cls_token,x],dim=1)
        x = x+self.position_embedding
        x = self.dropout(x)
        return x
class VisionTransformer(nn.Module):
    def __init__(self, in_shape,patch_size : tuple
                 ,num_heads,d_MLP,dropout,num_layers,classification=True,num_class:int = 2):
        super().__init__()
        in_channels,_,_ = in_shape
        embedding_dim = in_channels*patch_size[0]*patch_size[1]
        self.patchembedding = PatchEmbedding(in_shape,embedding_dim,patch_size,dropout)
        self.transformer = TransformerEncoder(embedding_dim,num_heads,d_MLP,dropout,num_layers)
        self.classification = classification
        if classification:
            self.MLP = nn.Sequential(
                nn.LayerNorm(embedding_dim),
                nn.Linear(embedding_dim,num_class)
            )

    def forward(self,x):
        b,c,h,w  = x.size()
        x = self.patchembedding(x)

        x = self.transformer(x)

        x = self.MLP(x[:,0,:]) if self.classification else x[:, 1:, :].view(b,c,h,w) 
        return x
        






if __name__ =='__main__':
    # a = torch.randn(5,64,224,224).cuda()
    # out = MultiHeadAttention(512,8)(a,a,a)
    # out =  TransformerEncoder(d_model=512,
    #                                num_heads=8,
    #                                d_MLP=2048,
    #                                dropout=0.2,
    #                                num_layers=12)(a)
    insize = (256,14,14)
    a = torch.randn((5,256,14,14)).cuda()
    # trans = VisionTransformer(
    #     in_shape=insize,
    #     patch_size=(1,1),
    #     num_heads=8,
    #     d_MLP = 512,
    #     dropout=0.2,
    #     num_layers=3,
    #     classification=False,
    #     num_class=10
    # ).cuda()
    trans = VisionTransformer(
            in_shape=(256,14,14),
            patch_size=(2,2),
            num_heads=8,
            d_MLP=1024,
            dropout=0.1,
            num_layers=3,
            classification=False
        ).cuda()
    summary(trans,insize)
    out = trans(a)
    print(out.shape)