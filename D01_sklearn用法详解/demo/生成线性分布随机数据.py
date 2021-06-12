from matplotlib import pyplot
from sklearn.datasets import make_regression

if __name__ == "__main__":
    # 生成随机数据
    X, y = make_regression(n_samples=100, n_features=1, noise=0.2)

    # 绘制散点图
    pyplot.scatter(X, y)
    pyplot.show()
