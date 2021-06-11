import collections
from math import log


def entropy(y, base=2):
    """计算随机变量Y的熵"""
    count = collections.Counter(y)
    ans = 0
    for freq in count.values():
        prob = freq / len(y)
        ans -= prob * log(prob, base)
    return ans
