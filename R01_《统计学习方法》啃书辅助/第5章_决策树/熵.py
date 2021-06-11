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


if __name__ == "__main__":
    X, Y = load_example()
    print(entropy(Y))  # H(D)
