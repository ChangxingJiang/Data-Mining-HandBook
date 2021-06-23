from scipy.misc import derivative


def newton_method_linear(func, args=(), error=1e-6, dx=1e-6):
    """一维牛顿法求f(x)=0的值

    :param func: 目标函数
    :param args: 参数列表
    :param error: 容差
    :param dx: 计算导数时使用的dx
    :return:
    """
    x0, y0 = 0, func(0, *args)
    while True:
        d = derivative(func, x0, args=args, dx=dx)  # 计算一阶导数
        x1 = x0 - y0 / d
        if abs(x1 - x0) < error:
            return x1
        x0, y0 = x1, func(x1, *args)


if __name__ == "__main__":
    def f(x, k):
        return (x - k) ** 3


    print(newton_method_linear(f, args=(2,)))  # 1.999998535982025
