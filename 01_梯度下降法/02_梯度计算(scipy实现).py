from scipy.misc import derivative


def foo1(x):
    return x ** 2


def foo2(arr):
    return ((arr[0] + 3) ** 2 + (arr[1] + 4) ** 2) / 2


def partial_derivative(func, arr, dx=1e-6):
    """计算n元函数在某点各个自变量的偏导数列表（梯度向量）

    :param func: [function] n元函数
    :param arr: [list/tuple] 目标点的自变量坐标
    :param dx: [int/float] 计算时x的增量
    :return: [list] 偏导数
    """
    dimension = len(arr)
    ans = []
    for i in range(dimension):
        def f(x):
            arr2 = list(arr)
            arr2[i] = x
            return func(arr2)

        ans.append(derivative(f, arr[i], dx=dx))
    return ans


if __name__ == "__main__":
    print(derivative(foo1, 3, dx=1e-6))  # 6
    print(partial_derivative(foo2, [0, 0]))  # [3.000000000419334, 3.9999999996709334]
