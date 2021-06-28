from sklearn.tree import DecisionTreeClassifier

from code.adaboost import AdaBoost

if __name__ == "__main__":
    dataset = [[[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]],
               [1, 1, 1, -1, -1, -1, 1, 1, 1, -1]]

    clf = AdaBoost(dataset[0], dataset[1], DecisionTreeClassifier(max_depth=1))
    correct = 0
    for ii in range(10):
        if clf.predict(dataset[0][ii]) == dataset[1][ii]:
            correct += 1
    print("预测正确率:", correct / 10)
