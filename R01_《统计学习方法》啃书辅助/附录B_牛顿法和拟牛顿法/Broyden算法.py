from code.newton_method import broyden_algorithm

if __name__ == "__main__":
    # [0]
    print(broyden_algorithm(lambda x: x[0] ** 2, 1, epsilon=1e-6))

    # [-3.0000000003324554, -3.9999999998511546]
    print(broyden_algorithm(lambda x: ((x[0] + 3) ** 2 + (x[1] + 4) ** 2) / 2, 2, epsilon=1e-6))
