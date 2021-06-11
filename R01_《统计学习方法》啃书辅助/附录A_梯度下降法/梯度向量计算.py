from code.gradient_descent import partial_derivative

if __name__ == "__main__":
    # [6.000000000838668]
    print(partial_derivative(lambda x: x[0] ** 2, [3]))

    # [3.000000000419334, 3.9999999996709334]
    print(partial_derivative(lambda x: ((x[0] + 3) ** 2 + (x[1] + 4) ** 2) / 2, [0, 0]))
