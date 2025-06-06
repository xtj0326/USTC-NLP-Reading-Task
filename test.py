import torch
from torch.autograd import Variable
from transformer import build_transformer
from decoder import subsequent_mask


def load_model(path, src_vocab_size, tgt_vocab_size, n_layers=2):
    """加载预训练模型
    
    Args:
        path: 模型路径
        src_vocab_size: 源词汇表大小
        tgt_vocab_size: 目标词汇表大小
        n_layers: 编码器和解码器的层数
        
    Returns:
        Transformer: 加载的模型
    """
    model = build_transformer(src_vocab_size, tgt_vocab_size, n_layers=n_layers)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """贪婪解码
    
    Args:
        model: Transformer模型
        src: 源序列
        src_mask: 源序列掩码
        max_len: 最大生成长度
        start_symbol: 起始符号
        
    Returns:
        Tensor: 解码结果
    """
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    
    for i in range(max_len - 1):
        tgt_mask = subsequent_mask(ys.size(1)).type_as(src.data)
        out = model.decode(memory, src_mask, 
                          Variable(ys), 
                          Variable(tgt_mask))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    
    return ys


def test_sorting(model_path):
    """测试数字排序任务
    
    Args:
        model_path: 模型路径
    """
    # 模型参数
    V = 20  # 词汇表大小
    n_layers = 2
    
    # 加载模型
    try:
        model = load_model(model_path, V, V, n_layers)
        print("模型加载成功")
    except:
        print("模型加载失败，请先运行train.py训练模型")
        return
    
    # 测试样例
    test_cases = [
        [1, 9, 3, 7, 5, 2, 8, 6, 4, 10],
        [1, 15, 12, 7, 3, 9, 18, 5, 11, 2],
        [1, 6, 6, 4, 2, 9, 8, 7, 3, 5]
    ]
    
    for i, test_data in enumerate(test_cases):
        print(f"\n测试样例 {i+1}:")
        
        # 转换为张量
        src = Variable(torch.LongTensor([test_data]))
        src_mask = Variable(torch.ones(1, 1, len(test_data)))
        
        # 解码
        result = greedy_decode(model, src, src_mask, max_len=len(test_data), start_symbol=1)
        
        # 计算正确答案
        expected = test_data.copy()
        expected[1:] = sorted(expected[1:])
        
        print(f"输入序列: {test_data}")
        print(f"模型输出: {result[0].tolist()}")
        print(f"正确答案: {expected}")
        
        # 检查正确性
        is_correct = result[0].tolist() == expected
        print(f"结果: {'正确' if is_correct else '错误'}")


if __name__ == "__main__":
    test_sorting('model/sort_task.pkl') 