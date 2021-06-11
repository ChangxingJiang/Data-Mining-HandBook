from code.naive_bayes import NaiveBayesAlgorithmWithSmoothing

if __name__ == "__main__":
    # 例4.2
    dataset = [[(1, "S"), (1, "M"), (1, "M"), (1, "S"), (1, "S"),
                (2, "S"), (2, "M"), (2, "M"), (2, "L"), (2, "L"),
                (3, "L"), (3, "M"), (3, "M"), (3, "L"), (3, "L")],
               [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]]
    naive_bayes = NaiveBayesAlgorithmWithSmoothing(*dataset)
    print(naive_bayes.predict([2, "S"]))
