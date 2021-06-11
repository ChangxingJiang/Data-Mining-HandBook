from code.gradient_descent import gradient_descent

if __name__ == "__main__":
    # [0.0]
    print(gradient_descent(lambda x: x[0] ** 2, 1, eta=0.1, epsilon=1e-6))

    # [-2.998150057576512, -3.997780069092481]
    print(gradient_descent(lambda x: ((x[0] + 3) ** 2 + (x[1] + 4) ** 2) / 2, 2, eta=0.1, epsilon=1e-6))
