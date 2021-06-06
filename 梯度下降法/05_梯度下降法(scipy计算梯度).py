from scipy.misc import derivative


def foo1(arr):
    return arr[0] ** 2


def foo2(arr):
    return ((arr[0] + 3) ** 2 + (arr[1] + 4) ** 2) / 2


def partial_derivative(func, arr, dx=1e-6):
    """计算n元函数在某点的偏导数

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


def gradient_descent(func, n, eta, epsilon, maximum=1000):
    """梯度下降法

    :param func: [function] n元目标函数
    :param n: [int] 目标函数元数
    :param eta: [int/float] 学习率
    :param epsilon: [int/float] 学习精度
    :param maximum: [int] 最大学习次数
    :return: [list] 结果点坐标
    """
    x0 = [2] * n  # 取自变量初值
    y0 = func(x0)  # 计算函数值
    for _ in range(maximum):
        print("第", _, "次迭代", ":", round(x0[0], 4))

        nabla = partial_derivative(func, x0)  # 计算梯度

        # print(x0, "nabla:", nabla)

        x1 = [x0[i] - eta * nabla[i] for i in range(n)]  # 迭代自变量
        y1 = func(x1)  # 计算函数值

        if abs(y1 - y0) < epsilon:  # 如果当前变化量小于学习精度，则结束学习
            return x1

        x0, y0 = x1, y1


if __name__ == "__main__":
    # print(foo2([0, 0]))  # 12.5
    # print(partial_derivative(foo2, [0, 0]))  # [3,4]
    print(gradient_descent(foo1, 1, eta=0.1, epsilon=1e-6))  # [-3,-4]
