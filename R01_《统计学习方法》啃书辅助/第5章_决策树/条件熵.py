from code.dicision_tree import conditional_entropy
from code.example import load_li_5_1

if __name__ == "__main__":
    X, Y = load_li_5_1()
    print(conditional_entropy([X[i][0] for i in range(len(X))], Y))  # H(D|X=x_1)
