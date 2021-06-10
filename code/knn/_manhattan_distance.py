def manhattan_distance(x1, x2):
    """计算曼哈顿距离

    :param x1: [tuple/list] 第1个向量
    :param x2: [tuple/list] 第2个向量
    :return: 曼哈顿距离
    """
    n_features = len(x1)
    return sum(abs(x1[i] - x2[i]) for i in range(n_features))
