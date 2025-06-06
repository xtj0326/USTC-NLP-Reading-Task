import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import numpy as np


class PositionalEncoding(nn.Module):
    """位置编码
    
    在序列中添加位置信息，使模型能够利用序列的顺序
    """
    
    def __init__(self, d_model, dropout, max_len=5000):
        """初始化位置编码
        
        使用正弦和余弦函数实现:
        PE(pos, 2i) = sin(pos/10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        
        Args:
            d_model: 模型维度
            dropout: dropout概率
            max_len: 最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * 
                            -(math.log(10000.0) / d_model))
        
        # 偶数位置使用sin，奇数位置使用cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 添加批次维度并注册为缓冲区（不作为模型参数）
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """前向传播，添加位置编码到输入嵌入
        
        Args:
            x: 输入嵌入 [batch, seq_len, d_model]
            
        Returns:
            Tensor: 带位置编码的嵌入
        """
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x) 