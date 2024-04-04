import numpy as np
import matplotlib.pyplot as plt

p1 = np.linspace(0, 1, 100)
p2 = np.linspace(0, 1, 100)

p1, p2 = np.meshgrid(p1, p2)

p3 = (1 - p1) * p2 + p1 * (1 - p2)
bias3 = np.abs(p3 - 0.5)
bias1 = np.abs(p1 - 0.5)
bias2 = np.abs(p2 - 0.5)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(p1, p2, bias3 - bias1)

plt.show()
