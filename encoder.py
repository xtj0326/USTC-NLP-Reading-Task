import torch.nn as nn
from utils import clone_layers, ResidualConnection, LayerNorm


class Encoder(nn.Module):
    # Transformer编码器
    # 多个相同层堆叠，每层包含自注意力和前馈网络
    
    def __init__(self, layer, n_layers):
        # layer: 单个编码器层
        # n_layers: 层数
        super(Encoder, self).__init__()
        self.layers = clone_layers(layer, n_layers)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        # x: 输入张量
        # mask: 掩码
        # 依次通过每一层，最后归一化
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    # 编码器层
    # 包含两个子层：
    # 1. 多头自注意力
    # 2. 前馈神经网络
    # 每个子层都有残差连接和层归一化
    
    def __init__(self, size, self_attn, feed_forward, dropout):
        # size: 模型维度
        # self_attn: 自注意力模块
        # feed_forward: 前馈网络
        # dropout: dropout比例
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clone_layers(ResidualConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        # x: 输入张量
        # mask: 掩码
        # 先经过自注意力层，再经过前馈网络层
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward) 