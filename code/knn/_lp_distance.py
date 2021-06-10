def lp_distance(p, x1, x2):
    """计算Lp距离

    :param p: [int] 参数p
    :param x1: [tuple/list] 第1个向量
    :param x2: [tuple/list] 第2个向量
    :return: Lp距离
    """
    n_features = len(x1)
    return pow(sum(pow(abs(x1[i] - x2[i]), p) for i in range(n_features)), 1 / p)
