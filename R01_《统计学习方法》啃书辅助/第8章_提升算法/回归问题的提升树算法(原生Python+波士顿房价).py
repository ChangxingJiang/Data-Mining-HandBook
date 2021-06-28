from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from code.adaboost import AdaBoostRegressor

if __name__ == "__main__":
    X, Y = load_boston(return_X_y=True)
    x1, x2, y1, y2 = train_test_split(X, Y, test_size=1 / 3, random_state=0)

    seg = AdaBoostRegressor(x1, y1, DecisionTreeRegressor(max_depth=1), M=50)
    r = sum((seg.predict(x2[i]) - y2[i]) ** 2 for i in range(len(x2)))
    print("平方误差损失:", r)
