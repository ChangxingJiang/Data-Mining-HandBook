from code.newton_method import bfgs_algorithm_with_sherman_morrison

if __name__ == "__main__":
    # [0]
    print(bfgs_algorithm_with_sherman_morrison(lambda x: x[0] ** 2, 1, epsilon=1e-6))

    # [-3.0000000000105342, -4.000000000014043]
    print(bfgs_algorithm_with_sherman_morrison(lambda x: ((x[0] + 3) ** 2 + (x[1] + 4) ** 2) / 2, 2, epsilon=1e-6))
