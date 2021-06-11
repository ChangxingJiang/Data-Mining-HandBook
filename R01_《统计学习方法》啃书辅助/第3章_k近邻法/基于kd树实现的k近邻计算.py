from code.knn import KDTreeKNN

if __name__ == "__main__":
    dataset = [[(3, 3), (4, 3), (1, 1)], [1, 1, -1]]  # 训练数据集
    knn = KDTreeKNN(dataset[0], dataset[1], k=2)
    print(knn.count((3, 4)))  # 1
