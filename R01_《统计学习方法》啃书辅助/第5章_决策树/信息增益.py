from code.dicision_tree import information_gain
from code.example import load_li_5_1

if __name__ == "__main__":
    X, Y = load_li_5_1()
    print(information_gain(X, Y, idx=0))  # g(D,A1)
