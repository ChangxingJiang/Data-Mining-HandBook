import time

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from code.svm import SVM

if __name__ == "__main__":
    X, Y = load_breast_cancer(return_X_y=True)
    for i in range(len(Y)):
        if Y[i] == 0:
            Y[i] = -1
    x1, x2, y1, y2 = train_test_split(X, Y, test_size=1 / 3, random_state=0)

    start_time = time.time()

    svm = SVM(x1, y1)
    n1, n2 = 0, 0
    for xx, yy in zip(x2, y2):
        if svm.predict(xx) == yy:
            n1 += 1
        else:
            n2 += 1

    end_time = time.time()

    print("正确率:", n1 / (n1 + n2))
    print("运行时间:", end_time - start_time)
