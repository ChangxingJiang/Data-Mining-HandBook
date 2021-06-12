from matplotlib import pyplot
from pandas import DataFrame
from sklearn.datasets import make_circles

if __name__ == "__main__":
    # 生成随机数据
    X, y = make_circles(n_samples=1000, noise=0.2)

    # 绘制散点图
    df = DataFrame({"x": X[:, 0], "y": X[:, 1], "label": y})
    colors = {0: "red", 1: "blue"}
    fig, ax = pyplot.subplots()
    grouped = df.groupby("label")
    for key, group in grouped:
        group.plot(ax=ax, kind="scatter", x="x", y="y", label=key, color=colors[key])
    pyplot.show()
