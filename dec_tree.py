from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn import datasets
import matplotlib.pylab as plt

iris = datasets.load_iris()
X_train = iris.data[5:147]
# print(X_train)

Y_train = iris.target[5:147]
print(Y_train)

X_test = iris.data[1]
y_test = iris.target[1]
print("actual: ", y_test)


# Building the classification model using a pre-defined parameter
dtc = DecisionTreeClassifier(max_depth=3)

# Train the model
dtc.fit(X_train, Y_train)

# Test the model
print("predicted: ", dtc.predict([X_test]))

print(dtc.decision_path([X_test], check_input=True))