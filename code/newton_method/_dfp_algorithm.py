import numpy as np

from ..gradient_descent import golden_section_for_line_search
from ..gradient_descent import partial_derivative


def dfp_algorithm(func, n_features, epsilon=1e-6, distance=3, maximum=1000):
    """DFP算法

    :param func: [function] n元目标函数
    :param n_features: [int] 目标函数元数
    :param epsilon: [int/float] 学习精度
    :param distance: [int/float] 每次一维搜索的长度范围（distance倍梯度的模）
    :param maximum: [int] 最大学习次数
    :return: [list] 结果点坐标
    """
    G0 = np.identity(n_features)  # 构造初始矩阵G0(单位矩阵)
    x0 = [0] * n_features  # 取初始点x0

    for _ in range(maximum):
        # 计算梯度 gk
        nabla = partial_derivative(func, x0)
        g0 = np.matrix([nabla]).T  # g0 = g_k

        # 当梯度的模长小于精度要求时，停止迭代
        if pow(sum([nabla[i] ** 2 for i in range(n_features)]), 0.5) < epsilon:
            return x0

        # 计算pk
        pk = - G0 * g0

        # 一维搜索求lambda_k
        def f(x):
            """pk 方向的一维函数"""
            x2 = [x0[j] + x * float(pk[j][0]) for j in range(n_features)]
            return func(x2)

        lk = golden_section_for_line_search(f, 0, distance, epsilon=1e-6)  # lk = lambda_k

        # 更新当前点坐标
        x1 = [x0[j] + lk * float(pk[j][0]) for j in range(n_features)]

        # 计算g_{k+1}，若模长小于精度要求时，则停止迭代
        nabla = partial_derivative(func, x1)
        g1 = np.matrix([nabla]).T  # g0 = g_{k+1}

        # 当梯度的模长小于精度要求时，停止迭代
        if pow(sum([nabla[i] ** 2 for i in range(n_features)]), 0.5) < epsilon:
            return x1

        # 计算G_{k+1}
        yk = g1 - g0
        dk = np.matrix([[lk * float(pk[j][0]) for j in range(n_features)]]).T

        G1 = G0 + (dk * dk.T) / (dk.T * yk) + (G0 * yk * yk.T * G0) / (yk.T * G0 * yk)

        G0 = G1
        x0 = x1
