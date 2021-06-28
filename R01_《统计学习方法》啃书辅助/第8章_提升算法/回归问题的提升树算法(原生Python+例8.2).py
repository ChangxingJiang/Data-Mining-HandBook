from sklearn.tree import DecisionTreeRegressor

from code.adaboost import AdaBoostRegressor

if __name__ == "__main__":
    dataset = [[[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]],
               [5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05]]

    seg = AdaBoostRegressor(dataset[0], dataset[1], DecisionTreeRegressor(max_depth=1), M=6)
    r = sum((seg.predict(dataset[0][i]) - dataset[1][i]) ** 2 for i in range(10))
    print("平方误差损失:", r)
