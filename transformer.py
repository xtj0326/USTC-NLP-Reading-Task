import torch
import torch.nn as nn
import copy
from attention import MultiHeadAttention
from encoder import Encoder, EncoderLayer
from decoder import Decoder, DecoderLayer
from feedforward import FeedForward
from positional import PositionalEncoding
from embedding import Embedding, Generator


class Transformer(nn.Module):
    # Transformer模型
    # 编码器-解码器结构
    
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        # encoder: 编码器
        # decoder: 解码器
        # src_embed: 输入嵌入层
        # tgt_embed: 输出嵌入层
        # generator: 输出生成器
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        # src: 输入序列
        # tgt: 目标序列
        # src_mask: 输入掩码
        # tgt_mask: 目标掩码
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        # src: 输入序列
        # src_mask: 输入掩码
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        # memory: 编码器输出
        # src_mask: 输入掩码
        # tgt: 目标序列
        # tgt_mask: 目标掩码
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


def build_transformer(src_vocab_size, tgt_vocab_size, n_layers=6, 
                    d_model=512, d_ff=2048, heads=8, dropout=0.1):
    # 构建完整Transformer模型
    # src_vocab_size: 输入词表大小
    # tgt_vocab_size: 输出词表大小
    # n_layers: 编码器和解码器层数
    # d_model: 模型维度
    # d_ff: 前馈网络隐藏层维度
    # heads: 多头注意力头数
    # dropout: dropout比例
    c = copy.deepcopy
    
    attn = MultiHeadAttention(heads, d_model)
    ff = FeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    
    model = Transformer(
        encoder=Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), n_layers),
        decoder=Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), n_layers),
        src_embed=nn.Sequential(Embedding(d_model, src_vocab_size), c(position)),
        tgt_embed=nn.Sequential(Embedding(d_model, tgt_vocab_size), c(position)),
        generator=Generator(d_model, tgt_vocab_size)
    )
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    return model 