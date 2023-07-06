import numpy as np
import matplotlib.pyplot as plt


# with open(filename) as f:
#     lines = f.readlines()


data = np.loadtxt('time.dat')
print(data)

fig, ax = plt.subplots()
ax.plot(data[:, 0], data[:, 1])
ax.set_xlabel('block size')
ax.set_ylabel('time (s)')
ax.grid()
plt.show()