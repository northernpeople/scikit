import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import matplotlib.pylab as plt

boston = datasets.load_boston()

# print(boston.feature_names)

X_train = boston.data[0:30]

# print(X_train)

y_train = boston.target[0:30]

print( y_train)

y_train_binary = []

for price in y_train:
    y_train_binary.append(price >= 20)

# plt.plot(X_train, y_train_binary)
# plt.show()

X_test = boston.data[28]
y_test = boston.target[28]

model = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')

model.fit(X_train, y_train_binary)

print(model.coef_)
print(model.predict([X_test]))
print(model.predict_proba([X_test]))

print(y_test)
