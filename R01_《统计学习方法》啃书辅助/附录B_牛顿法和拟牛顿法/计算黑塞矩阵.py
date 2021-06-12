from code.newton_method import get_hessian

if __name__ == "__main__":
    # [[0 ,  -37],
    #  [-37, 0]]
    print(get_hessian(lambda x: (x[0] ** 3) * (x[1] ** 2) - 3 * x[0] * (x[1] ** 3) - x[0] * x[1] + 1, [0, 2]))

    # [[6 ,  -4],
    #  [-4, -16]]
    print(get_hessian(lambda x: (x[0] ** 3) * (x[1] ** 2) - 3 * x[0] * (x[1] ** 3) - x[0] * x[1] + 1, [1, 1]))
