import torch
import torch.nn as nn
import numpy as np
from utils import clone_layers, ResidualConnection, LayerNorm


class Decoder(nn.Module):
    # Transformer解码器
    
    def __init__(self, layer, n_layers):
        # layer: 单个解码器层
        # n_layers: 层数
        super(Decoder, self).__init__()
        self.layers = clone_layers(layer, n_layers)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        # x: 目标序列
        # memory: 编码器输出
        # src_mask: 源序列掩码
        # tgt_mask: 目标序列掩码
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    # 解码器层
    # 包含三个子层：
    # 1. 掩码自注意力
    # 2. 编码器-解码器注意力
    # 3. 前馈神经网络
    
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        # size: 模型维度
        # self_attn: 自注意力模块
        # src_attn: 源注意力模块
        # feed_forward: 前馈网络
        # dropout: dropout比例
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clone_layers(ResidualConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        # x: 目标序列
        # memory: 编码器输出
        # src_mask: 源序列掩码
        # tgt_mask: 目标序列掩码
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, memory, memory, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    # 生成下三角掩码
    mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    return torch.from_numpy(mask) == 0 