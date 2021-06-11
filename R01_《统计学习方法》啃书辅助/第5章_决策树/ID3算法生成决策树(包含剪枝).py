from code.dicision_tree import DecisionTreeID3
from code.example import load_li_5_1

if __name__ == "__main__":
    X, Y = load_li_5_1()
    print(DecisionTreeID3(X, Y, labels=["年龄", "有工作", "有自己的房子", "信贷情况"], alpha=0.2))
    print(DecisionTreeID3(X, Y, labels=["年龄", "有工作", "有自己的房子", "信贷情况"], alpha=0.3))
