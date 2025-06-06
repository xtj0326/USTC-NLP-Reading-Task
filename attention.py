import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import clone_layers


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    attn_weights = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        attn_weights = dropout(attn_weights)
        
    return torch.matmul(attn_weights, value), attn_weights


class MultiHeadAttention(nn.Module):
    # 多头注意力机制
    # 将注意力机制做多次，每次关注不同特征，最后合并
    
    def __init__(self, h, d_model, dropout=0.1):
        # h: 头的数量
        # d_model: 模型维度
        # dropout: dropout比例
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        
        self.d_k = d_model // h
        self.h = h
        
        self.linears = clone_layers(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        # 前向传播
        if mask is not None:
            mask = mask.unsqueeze(1)
            
        batch_size = query.size(0)
        
        query, key, value = [
            linear(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for linear, x in zip(self.linears, (query, key, value))
        ]
        
        x, self.attn = attention(query, key, value, mask, self.dropout)
        
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        
        return self.linears[-1](x) 