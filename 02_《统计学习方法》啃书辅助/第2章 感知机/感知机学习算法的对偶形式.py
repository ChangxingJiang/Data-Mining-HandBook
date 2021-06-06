dataset = [[(3, 3), 1], [(4, 3), 1], [(1, 1), -1]]  # 训练数据集


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


def perceptron_primitive_form(data, eta):
    """感知机学习算法的对偶形式

    :param data: [tuple/list,int(1/-1)] 训练数据集 len(data[0]) = n
    :param eta: [int/float] 学习率
    :return: [list,int/float] 感知机模型的a(alpha)和b
    """
    size = len(data)  # 样本点数量
    a0, b0 = [0] * size, 0  # 选取初值a0(alpha),b0

    gram = count_gram([item[0] for item in data])  # 计算gram矩阵

    while True:  # 不断迭代直至没有误分类点
        for i in range(size):
            yi = data[i][1]

            val = 0
            for j in range(size):
                xj, yj = data[j]
                val += a0[j] * yj * gram[i][j]

            if (yi * (val + b0)) <= 0:
                a0[i] += eta
                b0 += eta * yi
                break
        else:
            return a0, b0


if __name__ == "__main__":
    print(perceptron_primitive_form(dataset, eta=1))  # ([2, 0, 5], -3)
