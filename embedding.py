import torch.nn as nn
import math


class Embedding(nn.Module):
    """词嵌入层
    
    将输入token转换为向量表示
    """
    
    def __init__(self, d_model, vocab_size):
        """初始化嵌入层
        
        Args:
            d_model: 模型维度
            vocab_size: 词汇表大小
        """
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        """前向传播
        
        将嵌入乘以sqrt(d_model)以缩放嵌入值
        
        Args:
            x: 输入序列 [batch, seq_len]
            
        Returns:
            Tensor: 嵌入向量 [batch, seq_len, d_model]
        """
        return self.embedding(x) * math.sqrt(self.d_model)


class Generator(nn.Module):
    """生成器
    
    将模型输出转换为下一个token的概率分布
    """
    
    def __init__(self, d_model, vocab_size):
        """初始化生成器
        
        Args:
            d_model: 模型维度
            vocab_size: 词汇表大小
        """
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """前向传播
        
        Args:
            x: 模型输出 [batch, seq_len, d_model]
            
        Returns:
            Tensor: 概率分布 [batch, seq_len, vocab_size]
        """
        return nn.functional.log_softmax(self.proj(x), dim=-1) 