from code.naive_bayes import NaiveBayesAlgorithmHashmap

if __name__ == "__main__":
    dataset = [[(1, "S"), (1, "M"), (1, "M"), (1, "S"), (1, "S"),
                (2, "S"), (2, "M"), (2, "M"), (2, "L"), (2, "L"),
                (3, "L"), (3, "M"), (3, "M"), (3, "L"), (3, "L")],
               [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]]
    naive_bayes = NaiveBayesAlgorithmHashmap(*dataset)
    print(naive_bayes.predict([2, "S"]))
