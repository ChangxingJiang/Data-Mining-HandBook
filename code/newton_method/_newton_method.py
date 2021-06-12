import numpy as np

from ._get_hessian import get_hessian
from ..gradient_descent import partial_derivative


def newton_method(func, n_features, epsilon=1e-6, maximum=1000):
    """牛顿法

    :param func: [function] n元目标函数
    :param n_features: [int] 目标函数元数
    :param epsilon: [int/float] 学习精度
    :param maximum: [int] 最大学习次数
    :return: [list] 结果点坐标
    """
    x0 = [0] * n_features  # 取初始点x0

    for _ in range(maximum):
        # 计算梯度 gk
        nabla = partial_derivative(func, x0)
        gk = np.matrix([nabla])

        # 当梯度的模长小于精度要求时，停止迭代
        if pow(sum([nabla[i] ** 2 for i in range(n_features)]), 0.5) < epsilon:
            return x0

        # 计算黑塞矩阵
        hessian = np.matrix(get_hessian(func, x0))

        # 计算步长 pk
        pk = - (hessian ** -1) * gk.T

        # 迭代
        for j in range(n_features):
            x0[j] += float(pk[j][0])
