def lp_distance(p, array1, array2):
    """计算Lp距离

    :param p: [int] 参数p
    :param array1: [tuple/list] 第1个向量
    :param array2: [tuple/list] 第2个向量
    :return:
    """
    n1, n2 = len(array1), len(array2)
    if n1 == n2:
        return pow(sum(pow(abs(array1[i] - array2[i]), p) for i in range(n1)), 1 / p)
    else:
        raise ValueError("向量维度数不一致")


if __name__ == "__main__":
    print(lp_distance(float("inf"), (0, 0), (1, 1)))  # 0
