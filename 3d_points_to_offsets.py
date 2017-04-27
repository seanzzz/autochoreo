import numpy as np
from sklearn import preprocessing

a = np.load("dance.npy")

print a
for i in reversed(range(1, a.shape[0])):
    for j in range(a.shape[1]):
        a[i, j] = a[i - 1, j] - a[i, j]

for j in range(a.shape[1]):
    a[0, j] = 0

min_max_scaler = preprocessing.MinMaxScaler()

a = min_max_scaler.fit_transform(a)

np.save("dance_offset_scaled", a)

print a
