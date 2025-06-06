# USTC-NLP-Reading-Task

# Transformer模型实现

本项目是一个精简版Transformer模型的实现，基于论文《Attention Is All You Need》。

## 项目结构

- `attention.py`: 实现注意力机制和多头注意力
- `decoder.py`: 实现Transformer解码器
- `embedding.py`: 实现词嵌入和生成器
- `encoder.py`: 实现Transformer编码器
- `feedforward.py`: 实现前馈神经网络
- `positional.py`: 实现位置编码
- `transformer.py`: 整合各组件构建完整Transformer模型
- `utils.py`: 实用工具函数和类
- `train.py`: 训练和测试脚本

## 示例任务

本项目实现了一个简单的数字排序任务，作为Transformer模型的演示：

1. 输入：随机整数序列
2. 输出：相同序列的排序版本（第一个数字保持不变，其余数字排序）

## 使用方法

```bash
# 训练模型
python train.py
```

## 主要特性

- 完整实现Transformer架构
- 多头注意力机制
- 位置编码
- 残差连接和层归一化
- 动态学习率调整
- 标签平滑

## 参数配置

可以在`train.py`的`main`函数中修改以下参数：

- `V`: 词汇表大小（最大数字+1）
- `batch_size`: 批大小
- `seq_len`: 序列长度
- `n_layers`: 编码器和解码器的层数

也可以在`build_transformer`函数中修改更多参数。 
