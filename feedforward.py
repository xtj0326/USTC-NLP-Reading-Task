import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """位置前馈网络
    
    由两个线性变换和一个ReLU激活函数组成
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        """初始化前馈网络
        
        FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
        
        Args:
            d_model: 模型维度
            d_ff: 隐藏层维度
            dropout: dropout概率
        """
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入张量 [batch, seq_len, d_model]
            
        Returns:
            Tensor: 输出张量 [batch, seq_len, d_model]
        """
        return self.w_2(self.dropout(F.relu(self.w_1(x)))) 