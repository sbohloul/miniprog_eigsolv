import sys
import numpy as np
import matplotlib.pyplot as plt


fname = sys.argv[1]

data = np.loadtxt(fname, skiprows=1)

x = data[:, 1]
y = data[:, 2]

plt.semilogy(x, y, 'o-')
plt.xlabel("n block")
plt.ylabel("time(s)")
plt.grid(True)
plt.show()