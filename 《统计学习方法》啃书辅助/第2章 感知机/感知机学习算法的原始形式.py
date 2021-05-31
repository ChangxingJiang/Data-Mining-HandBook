dataset = [[(3, 3), 1], [(4, 3), 1], [(1, 1), -1]]  # 训练数据集


def perceptron_primitive_form(data, eta):
    """感知机学习算法的原始形式

    :param data: [tuple/list,int(1/-1)] 训练数据集 len(data[0]) = n
    :param eta: [int/float] 学习率
    :return: [list,int/float] 感知机模型的w和b
    """
    size = len(data)  # 样本点数量
    n = len(data[0])  # 特征向量维度数
    w0, b0 = [0] * n, 0  # 选取初值w0,b0

    while True:  # 不断迭代直至没有误分类点
        for i in range(size - 1, -1, -1):
            xi, yi = data[i]
            if yi * (sum(w0[j] * xi[j] for j in range(n)) + b0) <= 0:
                w1 = [w0[j] + eta * yi * xi[j] for j in range(n)]
                b1 = b0 + eta * yi
                w0, b0 = w1, b1
                break
        else:
            return w0, b0


if __name__ == "__main__":
    print(perceptron_primitive_form(dataset, eta=1))
