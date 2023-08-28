from mpi4py import MPI
import pb11_time_scalapack_kernels
import numpy as np
import matplotlib.pyplot as plt
import h5py


# fname = 'Si_1040_HR.h5'
# f = h5py.File(fname, 'r')
# print(list(f.keys()))
# dset = f['hamiltonian']
# print(dset.keys())

# h = dset['total_gamma']
# print(h)
# h = h[:]
# print(type(h))

# o = dset['overlap_gamma']
# print(o)
# o = o[:]

# fig, axs = plt.subplots(1, 2)
# axs[0].spy(h, precision=1.0e-10)
# axs[1].spy(o, precision=1.0e-10)

# plt.show()

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

# ========================
# kernel launch parameters
# ========================
niter = 1
nprow = 2
npcol = 2
mb = 64
nb = 64

# ============
# Input arrays
# ============
fname = 'Si_1040_HR.h5'

if (mpi_rank == 0):
    f = h5py.File(fname, 'r')
    dset = f['hamiltonian']
    h = dset['total_gamma'][:]
    o = dset['overlap_gamma'][:]
    c = 0 * h
else:
    h = np.array([0], dtype=np.float64)
    o = np.array([0], dtype=np.float64)
    c = np.array([0], dtype=np.float64)

if mpi_rank == 0:
    m = h.shape[0]
    n = h.shape[1]
else:
    m = None
    n = None

m = mpi_comm.bcast(m, root=0)
n = mpi_comm.bcast(n, root=0)
if mpi_rank == 1:
    print(f"m = {m}, n = {n}")


t_kernel = pb11_time_scalapack_kernels.pb11_time_scalapack_pdgemm(
    niter, nprow, npcol, h, o, c, m, n, mb, nb)

t_kernel = mpi_comm.gather(t_kernel, root=0)

if mpi_rank == 0:
    print(t_kernel)
