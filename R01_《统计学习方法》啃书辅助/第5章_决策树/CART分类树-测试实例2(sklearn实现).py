from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text

if __name__ == "__main__":
    iris = load_iris()
    X = iris.data
    Y = iris.target

    x1, x2, y1, y2 = train_test_split(X, Y, test_size=1 / 3, random_state=0)

    clf = DecisionTreeClassifier(ccp_alpha=0.02, random_state=0)
    clf.fit(X, Y)
    print(export_text(clf, feature_names=iris.feature_names, show_weights=True))
    print(clf.score(x2, y2))
