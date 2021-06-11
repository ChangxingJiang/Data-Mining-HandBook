from ._golden_section_for_line_search import golden_section_for_line_search
from ._partial_derivative import partial_derivative


def steepest_descent(func, n_features, epsilon, distance=3, maximum=1000):
    """梯度下降法

    :param func: [function] n元目标函数
    :param n_features: [int] 目标函数元数
    :param epsilon: [int/float] 学习精度
    :param distance: [int/float] 每次一维搜索的长度范围（distance倍梯度的模）
    :param maximum: [int] 最大学习次数
    :return: [list] 结果点坐标
    """
    x0 = [0] * n_features  # 取自变量初值
    y0 = func(x0)  # 计算函数值
    for _ in range(maximum):
        nabla = partial_derivative(func, x0)  # 计算梯度

        # 当梯度的模长小于精度要求时，停止迭代
        if pow(sum([nabla[i] ** 2 for i in range(n_features)]), 0.5) < epsilon:
            return x0

        def f(x):
            """梯度方向的一维函数"""
            x2 = [x0[i] - x * nabla[i] for i in range(n_features)]
            return func(x2)

        lk = golden_section_for_line_search(f, 0, distance, epsilon=1e-6)  # 一维搜索寻找驻点

        x1 = [x0[i] - lk * nabla[i] for i in range(n_features)]  # 迭代自变量
        y1 = func(x1)  # 计算函数值

        if abs(y1 - y0) < epsilon:  # 如果当前变化量小于学习精度，则结束学习
            return x1

        x0, y0 = x1, y1
