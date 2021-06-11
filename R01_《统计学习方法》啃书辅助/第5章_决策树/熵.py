from code.dicision_tree import entropy
from code.example import load_li_5_1

if __name__ == "__main__":
    X, Y = load_li_5_1()
    print(entropy(Y))  # H(D)
