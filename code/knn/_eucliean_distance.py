def euclidean_distance(x1, x2):
    """计算欧氏距离

    :param x1: [tuple/list] 第1个向量
    :param x2: [tuple/list] 第2个向量
    :return: 欧氏距离
    """
    n_features = len(x1)
    return pow(sum(pow(x1[i] - x2[i], 2) for i in range(n_features)), 1 / 2)
