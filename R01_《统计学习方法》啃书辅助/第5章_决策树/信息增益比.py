from code.dicision_tree import information_gain_ratio
from code.example import load_li_5_1

if __name__ == "__main__":
    X, Y = load_li_5_1()
    print(information_gain_ratio(X, Y, idx=0))  # gR(D,A1)
    print(information_gain_ratio(X, Y, idx=1))  # gR(D,A2)
    print(information_gain_ratio(X, Y, idx=2))  # gR(D,A3)
    print(information_gain_ratio(X, Y, idx=3))  # gR(D,A4)
