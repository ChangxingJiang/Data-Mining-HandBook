import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text


def load_example():
    return [np.array([["青年", "否", "否", "一般"],
                      ["青年", "否", "否", "好"],
                      ["青年", "是", "否", "好"],
                      ["青年", "是", "是", "一般"],
                      ["青年", "否", "否", "一般"],
                      ["中年", "否", "否", "一般"],
                      ["中年", "否", "否", "好"],
                      ["中年", "是", "是", "好"],
                      ["中年", "否", "是", "非常好"],
                      ["中年", "否", "是", "非常好"],
                      ["老年", "否", "是", "非常好"],
                      ["老年", "否", "是", "好"],
                      ["老年", "是", "否", "好"],
                      ["老年", "是", "否", "非常好"],
                      ["老年", "否", "否", "一般"]]),
            np.array(["否", "否", "是", "是", "否",
                      "否", "否", "是", "是", "是",
                      "是", "是", "是", "是", "否"])]


if __name__ == "__main__":
    X, Y = load_example()

    N = len(X)
    n = len(X[0])

    # 坐标压缩（将可能存在的非数值的特征及类别转换为数值）
    y_list = list(set(Y))
    y_mapping = {c: i for i, c in enumerate(y_list)}
    x_list = [list(set(X[i][j] for i in range(N))) for j in range(n)]
    x_mapping = [{c: i for i, c in enumerate(x_list[j])} for j in range(n)]

    for i in range(N):
        for j in range(n):
            X[i][j] = x_mapping[j][X[i][j]]
    for i in range(N):
        Y[i] = y_mapping[Y[i]]

    clf = DecisionTreeClassifier()
    clf.fit(X, Y)
    print(export_text(clf, feature_names=["年龄", "有工作", "有自己的房子", "信贷情况"],show_weights=True))

