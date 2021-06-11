from code.dicision_tree import DecisionTreeC45WithoutPruning
from code.example import load_li_5_1

if __name__ == "__main__":
    X, Y = load_li_5_1()
    decision_tree = DecisionTreeC45WithoutPruning(X, Y, labels=["年龄", "有工作", "有自己的房子", "信贷情况"])
    print(decision_tree)
