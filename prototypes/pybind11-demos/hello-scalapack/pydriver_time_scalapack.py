from mpi4py import MPI
import _pb11_time_scalapack_kernels
import numpy as np

comm = MPI.COMM_WORLD
myrank = comm.Get_rank()
comm_size = comm.Get_size()

niter = 2
nprow = 2
npcol = 2
m = 2000
n = 2000
mb = 10
nb = 10

a = np.ones((1, m*n), dtype=np.float64)
b = 2 * np.ones((1, m*n), dtype=np.float64)
c = np.zeros((1, m*n), dtype=np.float64)

# if myrank == 0:
#     print("a: ")
#     print(a)
#     print("b: ")
#     print(b)
#     print("c: ")
#     print(c)
# comm.Barrier()

t_kernel = _pb11_time_scalapack_kernels.pb11_time_scalapack_pdgemm(
    niter, nprow, npcol, a, b, c, m, n, mb, nb)

c = comm.allreduce(c, op=MPI.SUM)
# if myrank == 1:
#     print(c)

for rank in range(comm_size):
    if myrank == rank:
        print("myrank: ", myrank, "t_kernel: ", t_kernel, flush=True)
    comm.Barrier()

t_kernel = comm.gather(t_kernel, root=0)
if myrank == 0:
    print(t_kernel)

# if myrank == 0:
#     print("a: ")
#     print(a)
#     print("b: ")
#     print(b)
#     print("c: ")
#     print(c)

# for irank in range(comm_size):
#     if (myrank == irank):
#         print("size: ", comm_size, " myrank: ", myrank)
#     comm.Barrier()
