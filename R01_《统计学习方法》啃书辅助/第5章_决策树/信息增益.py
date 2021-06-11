import collections
from math import log

import numpy as np


def load_example():
    return [np.array([["青年", "否", "否", "一般"],
                      ["青年", "否", "否", "好"],
                      ["青年", "是", "否", "好"],
                      ["青年", "是", "是", "一般"],
                      ["青年", "否", "否", "一般"],
                      ["中年", "否", "否", "一般"],
                      ["中年", "否", "否", "好"],
                      ["中年", "是", "是", "好"],
                      ["中年", "否", "是", "非常好"],
                      ["中年", "否", "是", "非常好"],
                      ["老年", "否", "是", "非常好"],
                      ["老年", "否", "是", "好"],
                      ["老年", "是", "否", "好"],
                      ["老年", "是", "否", "非常好"],
                      ["老年", "否", "否", "一般", "否"]]),
            np.array(["否", "否", "是", "是", "否",
                      "否", "否", "是", "是", "是",
                      "是", "是", "是", "是", "否"])]


def entropy(y, base=2):
    """计算随机变量Y的熵"""
    count = collections.Counter(y)
    ans = 0
    for freq in count.values():
        prob = freq / len(y)
        ans -= prob * log(prob, base)
    return ans


def conditional_entropy(x, y, base=2):
    """计算随机变量X给定的条件下随机变量Y的条件熵H(Y|X)"""
    freq_y_total = collections.defaultdict(collections.Counter)  # 统计随机变量X取得每一个取值时随机变量Y的频数
    freq_x = collections.Counter()  # 统计随机变量X每一个取值的频数
    for i in range(len(x)):
        freq_y_total[x[i]][y[i]] += 1
        freq_x[x[i]] += 1
    ans = 0
    for xi, freq_y_xi in freq_y_total.items():
        res = 0
        for freq in freq_y_xi.values():
            prob = freq / freq_x[xi]
            res -= prob * log(prob, base)
        ans += res * (freq_x[xi] / len(x))
    return ans


def information_gain(x, y, idx, base=2):
    """计算特征A(第idx个特征)对训练数据集D(输入数据x,输出数据y)的信息增益"""
    return entropy(y, base=base) - conditional_entropy([x[i][idx] for i in range(len(x))], y, base=base)


if __name__ == "__main__":
    X, Y = load_example()
    print(information_gain(X, Y, idx=0))  # g(D,A1)
