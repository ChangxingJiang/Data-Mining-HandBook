from code.perceptron import original_form_of_perceptron

if __name__ == "__main__":
    dataset = [[(3, 3), (4, 3), (1, 1)], [1, 1, -1]]  # 训练数据集
    print(original_form_of_perceptron(dataset[0], dataset[1], eta=1))
