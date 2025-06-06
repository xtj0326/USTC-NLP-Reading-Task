import os
import torch
import numpy as np
from torch.autograd import Variable
from transformer import build_transformer
from utils import LabelSmoothing, NoamOptimizer, run_epoch
from decoder import subsequent_mask


class Batch:
    """批次数据类，用于处理输入序列和目标序列"""
    def __init__(self, src, tgt=None, pad=0):
        self.src = src  # 源序列
        self.src_mask = (src != pad).unsqueeze(-2)  # 源序列掩码
        
        if tgt is not None:
            self.tgt = tgt[:, :-1]  # 目标序列（去掉最后一个token）
            self.tgt_y = tgt[:, 1:]  # 目标序列（去掉第一个token）
            self.tgt_mask = self.make_std_mask(self.tgt, pad)  # 目标序列掩码
            self.ntokens = (self.tgt_y != pad).data.sum().item()  # 非填充token数量

    @staticmethod
    def make_std_mask(tgt, pad):
        """创建标准掩码，用于解码器"""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


def generate_sorted_numbers(V, batch_size, seq_len, nbatches):
    """生成排序任务的数据
    
    生成随机数字序列，第一个数字固定为1，其余数字需要排序
    """
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch_size, seq_len)))
        data[:, 0] = 1  # 第一个数字固定为1
        
        tgt = torch.zeros_like(data)
        for b in range(batch_size):
            sorted_seq = data[b].clone()
            sorted_seq[1:] = torch.sort(data[b][1:])[0]  # 对除第一个数字外的序列进行排序
            tgt[b] = sorted_seq
            
        src = Variable(data, requires_grad=False)
        tgt = Variable(tgt, requires_grad=False)
        
        yield Batch(src, tgt, 0)


class SimpleLossCompute:
    """简单的损失计算类"""
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator  # 生成器
        self.criterion = criterion  # 损失函数
        self.opt = opt  # 优化器

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                             y.contiguous().view(-1)) / norm
        
        loss.backward()  # 反向传播
        
        if self.opt is not None:
            self.opt.step()  # 更新参数
            self.opt.optimizer.zero_grad()  # 清空梯度
            
        return loss.item() * norm


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """贪婪解码函数
    
    使用贪婪策略生成输出序列
    """
    memory = model.encode(src, src_mask)  # 编码输入序列
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)  # 初始化输出序列
    
    for i in range(max_len - 1):
        tgt_mask = subsequent_mask(ys.size(1)).type_as(src.data)
        out = model.decode(memory, src_mask, 
                          Variable(ys), 
                          Variable(tgt_mask))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)  # 选择概率最大的词
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    
    return ys


def main():
    """主函数：训练和测试模型"""
    # 模型参数
    V = 20  # 词汇表大小
    batch_size = 30  # 批次大小
    seq_len = 10  # 序列长度
    n_layers = 2  # 模型层数
    
    model = build_transformer(V, V, n_layers=n_layers)
    
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    
    model_opt = NoamOptimizer(model.src_embed[0].d_model, 1, 400,
                          torch.optim.Adam(model.parameters(), lr=0, 
                                          betas=(0.9, 0.98), eps=1e-9))
    
    print("开始训练...")
    best_loss = float('inf')
    for epoch in range(20):
        model.train()
        train_loss = run_epoch(
            generate_sorted_numbers(V, batch_size, seq_len, 20),
            model, 
            SimpleLossCompute(model.generator, criterion, model_opt)
        )
        
        model.eval()
        val_loss = run_epoch(
            generate_sorted_numbers(V, batch_size, seq_len, 5),
            model, 
            SimpleLossCompute(model.generator, criterion, None)
        )
        
        print(f"Epoch {epoch+1}, 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            if not os.path.exists("model"):
                os.mkdir("model")
            torch.save(model.state_dict(), 'model/sort_task.pkl')
            print("保存最佳模型")
    
    print("\n测试模型...")
    model.eval()
    
    test_data = torch.LongTensor([[1, 15, 8, 6, 12, 4, 9, 18, 2, 10]])
    print(f"输入序列: {test_data[0].tolist()}")
    
    src = Variable(test_data)
    src_mask = Variable(torch.ones(1, 1, test_data.size(1)))
    result = greedy_decode(model, src, src_mask, max_len=test_data.size(1), start_symbol=1)
    
    print(f"模型输出: {result[0].tolist()}")
    
    expected = test_data.clone()
    expected[0, 1:] = torch.sort(test_data[0, 1:])[0]
    print(f"正确答案: {expected[0].tolist()}")


if __name__ == "__main__":
    main()