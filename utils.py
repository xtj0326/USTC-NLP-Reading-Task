import copy
import torch
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable


def clone_layers(module, n):
    """生成n个相同的层
    
    Args:
        module: 要复制的模块
        n: 复制次数
        
    Returns:
        nn.ModuleList: 复制生成的模块列表
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class LayerNorm(nn.Module):
    """层归一化"""
    
    def __init__(self, features, eps=1e-6):
        """初始化层归一化
        
        将每个子层的输出归一化: LayerNorm(x+Sublayer(x))
        
        Args:
            features: 特征维度
            eps: 防止除零的小常数
        """
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        """前向传播"""
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class ResidualConnection(nn.Module):
    """残差连接模块"""
    
    def __init__(self, size, dropout):
        """初始化残差连接
        
        对应论文的 Add & Norm
        
        Args:
            size: 模型维度
            dropout: dropout概率
        """
        super(ResidualConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """前向传播，实现残差连接
        
        将输入与经过子层处理后的输出相加
        """
        return x + self.dropout(sublayer(self.norm(x)))


def run_epoch(data_iter, model, loss_compute):
    """训练或评估一个epoch
    
    Args:
        data_iter: 数据迭代器
        model: 模型
        loss_compute: 损失计算函数
        
    Returns:
        float: 平均损失
    """
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss = loss_compute(out, batch.tgt_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print(f"步骤: {i} 损失: {loss / batch.ntokens:.4f} "
                  f"速度: {tokens / elapsed:.2f} tokens/sec")
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


class NoamOptimizer:
    """Noam优化器，实现学习率调度"""
    
    def __init__(self, model_size, factor, warmup, optimizer):
        """初始化Noam优化器
        
        Args:
            model_size: 模型维度
            factor: 缩放因子
            warmup: 预热步数
            optimizer: 基础优化器
        """
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        """更新参数和学习率"""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """计算学习率
        
        Args:
            step: 当前步数，默认使用内部步数
        
        Returns:
            float: 学习率
        """
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * 
                             min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_optimizer(model):
    """获取标准优化器
    
    Args:
        model: 模型
        
    Returns:
        NoamOptimizer: 优化器实例
    """
    return NoamOptimizer(
        model.src_embed[0].d_model, 2, 4000,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    )


class LabelSmoothing(nn.Module):
    """标签平滑"""
    
    def __init__(self, size, padding_idx, smoothing=0.0):
        """初始化标签平滑
        
        防止模型过度自信，使用KL散度损失实现
        
        Args:
            size: 词汇表大小
            padding_idx: 填充符索引
            smoothing: 平滑系数
        """
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        """前向传播
        
        Args:
            x: 预测分布(log概率)
            target: 真实标签
            
        Returns:
            Tensor: 损失值
        """
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1).long(), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False)) 