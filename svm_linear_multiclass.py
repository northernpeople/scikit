from sklearn import svm
import numpy as np
from sklearn import datasets
import matplotlib.pylab as plt

boston = datasets.load_boston()
X_train = boston.data[0:30]
# print(X_train)

y_train = boston.target[0:30]
# print(boston.target)

y_train_3 = []

for price in y_train:
    description = "MID"
    if price >= 25:
        description = "UPPER"
    if price <= 19:
        description = "LOWER"
    y_train_3.append(description)

# plt.plot(y_train_3)
# plt.show()

X_test = boston.data[28]
y_test = boston.target[28]

print(y_test)


X_test1 = boston.data[35]
y_test1 = boston.target[35]

print(y_test1)


X_test2 = boston.data[3]
y_test2 = boston.target[3]

print(y_test2)

clf = svm.SVC(gamma='scale', decision_function_shape='ovo')
clf.fit(X_train, y_train_3)

print(clf.predict([X_test]))
print(clf.predict([X_test1]))
print(clf.predict([X_test2]))


# # get support vectors
# print(clf.support_vectors_)
#
# # get indices of support vectors
# print(clf.support_)
#
# # get number of support vectors for each class
# print(clf.n_support_)
