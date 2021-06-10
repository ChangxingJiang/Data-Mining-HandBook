from code.perceptron import dual_form_perceptron

if __name__ == "__main__":
    dataset = [[(3, 3), (4, 3), (1, 1)], [1, 1, -1]]  # 训练数据集
    print(dual_form_perceptron(dataset[0], dataset[1], eta=1))  # ([2, 0, 5], -3)
