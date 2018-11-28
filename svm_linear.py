from sklearn import svm
import numpy as np
from sklearn import datasets
import matplotlib.pylab as plt

boston = datasets.load_boston()
X_train = boston.data[0:30]
# print(X_train)

y_train = boston.target[0:30]
# print( y_train)

y_train_binary = []

for price in y_train:
    y_train_binary.append(price >= 20)

# plt.plot(X_train, y_train_binary)
# plt.show()

X_test = boston.data[28]
y_test = boston.target[28]

print(y_test)

clf = svm.SVC()
clf.fit(X_train, y_train_binary)

print(clf.predict([X_test]))

# # get support vectors
# print(clf.support_vectors_)
#
#
# # get indices of support vectors
# print(clf.support_)
#
# # get number of support vectors for each class
print(clf.n_support_)
