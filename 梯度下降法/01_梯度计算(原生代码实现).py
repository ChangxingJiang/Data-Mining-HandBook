# from scipy.misc import derivative


def foo1(x):
    return x ** 2


def foo2(arr):
    return ((arr[0] + 3) ** 2 + (arr[1] + 4) ** 2) / 2


def derivative(func, x, dx=1e-6):
    """计算一元函数在某点的导数

    :param func: [function] 一元函数
    :param x: [int/float] 目标点的自变量坐标
    :param dx: [int/float] 计算时x的增量
    """
    return (func(x + dx) - func(x)) / dx


def partial_derivative(func, arr, dx=1e-6):
    """计算n元函数在某点的所有自变量的偏导数列表（梯度向量）

    :param func: [function] n元函数
    :param arr: [list/tuple] 目标点的自变量坐标
    :param dx: [int/float] 计算时x的增量
    :return: [list] 偏导数
    """
    ans = []
    for i in range(len(arr)):
        arr2 = list(arr)
        arr2[i] += dx
        ans.append((func(arr2) - func(arr)) / dx)
    return ans


if __name__ == "__main__":
    print(derivative(foo1, 3))  # 6
    print(partial_derivative(foo2, [0, 0]))  # [3.000000500463784, 4.000000499715384]
