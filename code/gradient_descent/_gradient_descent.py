from ._partial_derivative import partial_derivative


def gradient_descent(func, n_features, eta, epsilon, maximum=1000):
    """梯度下降法

    :param func: [function] n元目标函数
    :param n_features: [int] 目标函数元数
    :param eta: [int/float] 学习率
    :param epsilon: [int/float] 学习精度
    :param maximum: [int] 最大学习次数
    :return: [list] 结果点坐标
    """
    x0 = [0] * n_features  # 取自变量初值
    y0 = func(x0)  # 计算函数值
    for _ in range(maximum):
        nabla = partial_derivative(func, x0)  # 计算梯度
        x1 = [x0[i] - eta * nabla[i] for i in range(n_features)]  # 迭代自变量
        y1 = func(x1)  # 计算函数值
        if abs(y1 - y0) < epsilon:  # 如果当前变化量小于学习精度，则结束学习
            return x1
        x0, y0 = x1, y1
