import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt

a = np.load("model-43in30out-256cells2layers60steps_output.npy")
a = np.delete(a, list(xrange(30, 43)), axis=1)

a = np.reshape(a, (-1, 10, 3))
a = np.delete(a, 2, axis=2)
a = np.reshape(a, (-1, 20))

print a.shape


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
