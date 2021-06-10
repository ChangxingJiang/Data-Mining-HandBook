def count_gram(x):
    """计算Gram矩阵

    :param x: 输入变量
    :return: 输入变量的Gram矩阵
    """
    n_samples = len(x)  # 样本点数量
    n_features = len(x[0])  # 特征向量维度数
    gram = [[0] * n_samples for _ in range(n_samples)]  # 初始化Gram矩阵

    # 计算Gram矩阵
    for i in range(n_samples):
        for j in range(i, n_samples):
            gram[i][j] = gram[j][i] = sum(x[i][k] * x[j][k] for k in range(n_features))

    return gram
