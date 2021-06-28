from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from code.adaboost import AdaBoost

if __name__ == "__main__":
    X, Y = load_breast_cancer(return_X_y=True)
    x1, x2, y1, y2 = train_test_split(X, Y, test_size=1 / 3, random_state=0)

    clf = AdaBoost(x1, y1, DecisionTreeClassifier(max_depth=1))
    correct = 0
    for i in range(len(x2)):
        if clf.predict(x2[i]) == y2[i]:
            correct += 1
    print("预测正确率:", correct / len(x2))
