import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from sklearn import preprocessing

a = np.load("output-43in30out-999_scaled.npy")

dance = np.load("dance.npy")

origin = np.load("dance_offset.npy")

min_max_scaler = preprocessing.MinMaxScaler()

aa = min_max_scaler.fit_transform(origin)

a = min_max_scaler.inverse_transform(a)

print a

exit()

a = np.insert(a, 0, dance[120], axis=0)

for i in range(1, a.shape[0]):
    for j in range(a.shape[1]):
        a[i, j] = a[i - 1, j] + a[i, j]

a = np.delete(a, list(xrange(30, 43)), axis=1)

a = np.reshape(a, (-1, 10, 3))
a = np.delete(a, 2, axis=2)
a = np.reshape(a, (-1, 20))


def update_plot(i, fig, scat):
    scat.set_offsets(a[i])
    return scat


fig = plt.figure()

x = [0, 0, 0]
y = [0, 0, 0]

ax = fig.add_subplot(111)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

scat = plt.scatter(x, y, c=x)

anim = animation.FuncAnimation(
    fig, update_plot, fargs=(fig, scat), interval=1000 / 30)

plt.show()
