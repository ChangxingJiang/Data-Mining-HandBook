def count_gram(x):
    """计算Gram矩阵

    :param x: 输入变量
    :return: 输入变量的Gram矩阵
    """
    size = len(x)  # 样本点数量
    n = len(x[0])  # 特征向量维度数
    gram = [[0] * size for _ in range(size)]  # 初始化Gram矩阵

    # 计算Gram矩阵
    for i in range(size):
        for j in range(i, size):
            gram[i][j] = gram[j][i] = sum(x[i][k] * x[j][k] for k in range(n))

    return gram


def perceptron_primitive_form(x, y, eta):
    """感知机学习算法的对偶形式

     :param x: 输出变量
    :param y: 输出变量
    :param eta: 学习率
    :return: 感知机模型的a(alpha)和b
    """
    if len(x) != len(y):
        raise ValueError("输入变量和输出变量数量不同")

    size = len(x)  # 样本点数量
    a0, b0 = [0] * size, 0  # 选取初值a0(alpha),b0

    gram = count_gram(x)  # 计算gram矩阵

    while True:  # 不断迭代直至没有误分类点
        for i in range(size):
            yi = y[i]

            val = 0
            for j in range(size):
                xj, yj = x[j], y[j]
                val += a0[j] * yj * gram[i][j]

            if (yi * (val + b0)) <= 0:
                a0[i] += eta
                b0 += eta * yi
                break
        else:
            return a0, b0


if __name__ == "__main__":
    dataset = [[(3, 3), (4, 3), (1, 1)], [1, 1, -1]]  # 训练数据集
    print(perceptron_primitive_form(dataset[0], dataset[1], eta=1))  # ([2, 0, 5], -3)
