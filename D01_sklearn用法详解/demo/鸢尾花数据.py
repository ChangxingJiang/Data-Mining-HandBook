from matplotlib import pyplot
from pandas import DataFrame
from sklearn.datasets import load_iris

if __name__ == "__main__":
    # 生成随机数据
    X, y = load_iris(return_X_y=True)

    # 绘制散点图
    df = DataFrame({"x": X[:, 0], "y": X[:, 1], "label": y})
    colors = {0: "red", 1: "blue", 2: "gold", 3: "purple", 4: "green"}
    fig, ax = pyplot.subplots()
    grouped = df.groupby("label")
    for key, group in grouped:
        group.plot(ax=ax, kind="scatter", x="x", y="y", label=key, color=colors[key])
    pyplot.show()
