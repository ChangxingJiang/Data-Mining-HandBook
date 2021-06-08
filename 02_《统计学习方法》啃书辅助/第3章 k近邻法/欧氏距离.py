def euclidean_distance(array1, array2):
    """计算欧氏距离

    :param array1: [tuple/list] 第1个向量
    :param array2: [tuple/list] 第2个向量
    :return:
    """
    n1, n2 = len(array1), len(array2)
    if n1 == n2:
        return pow(sum(pow(array1[i] - array2[i], 2) for i in range(n1)), 1 / 2)
    else:
        raise ValueError("向量维度数不一致")


if __name__ == "__main__":
    print(euclidean_distance((0, 0), (1, 1)))  # 1.4142
