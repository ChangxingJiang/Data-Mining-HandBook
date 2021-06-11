from code.gradient_descent import steepest_descent

if __name__ == "__main__":
    # [0]
    print(steepest_descent(lambda x: x[0] ** 2, 1, epsilon=1e-6))

    # [-2.9999999999635865, -3.999999999951452]
    print(steepest_descent(lambda x: ((x[0] + 3) ** 2 + (x[1] + 4) ** 2) / 2, 2, epsilon=1e-6))
