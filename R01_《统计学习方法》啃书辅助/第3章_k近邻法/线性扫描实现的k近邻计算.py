from code.knn import LinearSweepKNN
from code.knn import euclidean_distance

if __name__ == "__main__":
    dataset = [[(3, 3), (4, 3), (1, 1)], [1, 1, -1]]  # 训练数据集
    knn = LinearSweepKNN(dataset[0], dataset[1], k=2, distance_func=euclidean_distance)
    print(knn.count((3, 4)))  # 1
