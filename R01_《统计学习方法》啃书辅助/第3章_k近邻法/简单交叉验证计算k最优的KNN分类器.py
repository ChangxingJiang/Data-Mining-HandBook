from sklearn.datasets import make_blobs

from code.knn import build_best_knn_simple_cross_validation

if __name__ == "__main__":
    # 生成随机样本数据
    X, Y = make_blobs(n_samples=1000,
                      n_features=10,
                      centers=5,
                      cluster_std=5000,
                      center_box=(-10000, 10000),
                      random_state=0)

    # 计算k最优的KNN分类器
    final_k, final_score = build_best_knn_simple_cross_validation(X, Y)

    print("最优k:", final_k)  # 75
    print("最优k的测试集准确率:", final_score)  # 0.900
