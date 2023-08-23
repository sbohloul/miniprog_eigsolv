# import os
# import sys

# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# parent_dir = parent_dir + "/lib"
# sys.path.append(parent_dir)

import pb11_time_blas_kernels as tk
import numpy as np
import matplotlib.pyplot as plt

niter = 100
stepsize = 10000
maxsize = 1000000

print("niter \t nelem \t time(s)")

t_kernel = []
nelem = []
for n in range(stepsize, maxsize, stepsize):
    nelem.append(n)
    x = np.ones((1, n), dtype=np.float64)
    y = np.ones((1, n), dtype=np.float64)
    t = tk.pb11_time_blas_ddot(niter, x, y)
    t_kernel.append(t)
    print(niter, "\t", n, "\t", t)

plt.semilogy(nelem, t_kernel, '-o')
plt.xlabel('nelem')
plt.ylabel('time(s)')
plt.grid(True)
plt.show()
