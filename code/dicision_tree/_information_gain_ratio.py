from ._entropy import entropy
from ._information_gain import information_gain


def information_gain_ratio(x, y, idx, base=2):
    """计算特征A(第idx个特征)对训练数据集D(输入数据x,输出数据y)的信息增益比"""
    return information_gain(x, y, idx, base=base) / entropy([x[i][idx] for i in range(len(x))], base=base)
