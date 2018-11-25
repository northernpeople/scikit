from sklearn import datasets
import numpy as np


def dist(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


digits = datasets.load_digits()

x_features = digits.data[0:1000]
x_labels = digits.target[0:1000]

x_test = digits.data[555]


num = len(x_features)
distances = np.zeros(num)

for i in range(num):
    distances[i] = dist(x_features[i], x_test)

min_index = np.argmin(distances)

print(x_labels[min_index])