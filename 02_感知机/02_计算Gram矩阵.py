def count_gram(data):
    """计算Gram矩阵

    :param array: [tuple/list] 训练数据集的输入变量列表（有序）
    :return: [list[float]] 训练数据集的输入变量的Gram矩阵
    """
    size = len(data)  # 样本点数量
    n = len(data[0])  # 特征向量维度数
    gram = [[0] * size for _ in range(size)]  # 初始化Gram矩阵

    # 计算Gram矩阵
    for i in range(size):
        for j in range(i, size):
            gram[i][j] = gram[j][i] = sum(data[i][k] * data[j][k] for k in range(n))

    return gram
