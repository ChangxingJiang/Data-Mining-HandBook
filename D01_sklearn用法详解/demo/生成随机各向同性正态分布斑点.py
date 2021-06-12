from matplotlib import pyplot
from pandas import DataFrame
from sklearn.datasets import make_blobs

if __name__ == "__main__":
    # 生成随机数据
    X, y = make_blobs(n_samples=500, n_features=10, centers=5,
                      cluster_std=5000, center_box=(-10000, 10000), random_state=0)

    # 绘制散点图
    df = DataFrame({"x": X[:, 0], "y": X[:, 1], "label": y})
    colors = {0: "red", 1: "blue", 2: "green", 3: "purple", 4: "gold"}
    fig, ax = pyplot.subplots()
    grouped = df.groupby("label")
    for key, group in grouped:
        group.plot(ax=ax, kind="scatter", x="x", y="y", label=key, color=colors[key])
    pyplot.show()
