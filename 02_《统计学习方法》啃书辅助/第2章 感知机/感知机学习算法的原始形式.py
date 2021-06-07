def perceptron_primitive_form(x, y, eta):
    """感知机学习算法的原始形式

    :param x: 输入变量
    :param y: 输出变量
    :param eta: 学习率
    :return: 感知机模型的w和b
    """
    if len(x) != len(y):
        raise ValueError("输入变量和输出变量数量不同")

    size = len(x)  # 样本点数量
    n = len(x[0])  # 特征向量维度数
    w0, b0 = [0] * n, 0  # 选取初值w0,b0

    while True:  # 不断迭代直至没有误分类点
        for i in range(size):
            xi, yi = x[i], y[i]
            if yi * (sum(w0[j] * xi[j] for j in range(n)) + b0) <= 0:
                w1 = [w0[j] + eta * yi * xi[j] for j in range(n)]
                b1 = b0 + eta * yi
                w0, b0 = w1, b1
                break
        else:
            return w0, b0


if __name__ == "__main__":
    dataset = [[(3, 3), (4, 3), (1, 1)], [1, 1, -1]]  # 训练数据集
    print(perceptron_primitive_form(dataset[0], dataset[1], eta=1))