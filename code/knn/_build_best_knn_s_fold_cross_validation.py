from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def build_best_knn_s_fold_cross_validation(x, y):
    """S折交叉验证计算k最优的KNN分类器"""
    x1, x2, y1, y2 = train_test_split(x, y, test_size=0.2, random_state=0)  # 拆分训练集(80%)和测试集(20%)
    best_k, best_score = 0, 0
    for k in range(1, 101):
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, x1, y1, cv=10, scoring="accuracy")
        score = scores.mean()
        if score > best_score:
            best_k, best_score = k, score
    best_knn = KNeighborsClassifier(n_neighbors=best_k)
    best_knn.fit(x1, y1)
    return best_k, best_knn.score(x2, y2)
