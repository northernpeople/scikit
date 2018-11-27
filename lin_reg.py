import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets

boston = datasets.load_boston()

# print(boston.feature_names)

X_train = boston.data[0:100]

# print(X_train)

y_train = boston.target[0:100]

X_test = boston.data[15]
y_test = boston.target[15]

model = LinearRegression()
model.fit(X_train, y_train)

print(model.predict([X_test]))

print(y_test)
