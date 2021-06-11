from ._conditional_extropy import conditional_entropy
from ._entropy import entropy


def information_gain(x, y, idx, base=2):
    """计算特征A(第idx个特征)对训练数据集D(输入数据x,输出数据y)的信息增益"""
    return entropy(y, base=base) - conditional_entropy([x[i][idx] for i in range(len(x))], y, base=base)
