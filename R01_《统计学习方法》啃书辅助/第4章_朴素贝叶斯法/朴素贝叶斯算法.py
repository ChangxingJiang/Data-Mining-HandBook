from code.naive_bayes import NaiveBayesAlgorithmArray
from code.naive_bayes import NaiveBayesAlgorithmHashmap

if __name__ == "__main__":
    dataset = [[(1, "S"), (1, "M"), (1, "M"), (1, "S"), (1, "S"),
                (2, "S"), (2, "M"), (2, "M"), (2, "L"), (2, "L"),
                (3, "L"), (3, "M"), (3, "M"), (3, "L"), (3, "L")],
               [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]]
    naive_bayes_1 = NaiveBayesAlgorithmHashmap(*dataset)
    print(naive_bayes_1.predict([2, "S"]))

    naive_bayes_2 = NaiveBayesAlgorithmArray(*dataset)
    print(naive_bayes_2.predict([2, "S"]))
