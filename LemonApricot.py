
from sklearn import tree

# Data to train the machine
# features = [[140, "smooth"], [130, "smooth"], [150, "bumpy"], [170, "bumpy"]]
features = [[140, 1], [130, 1], [150, 0], [170, 0]]
# labels = ["apricot", "apricot", "lemon", "lemon"]
labels = [0, 0, 1, 1]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

print(clf.predict([[160, 0]]))
# Output: 0-apricot, 1-lemon
# Correct output is: 1-lemon