import collections
from math import log


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
