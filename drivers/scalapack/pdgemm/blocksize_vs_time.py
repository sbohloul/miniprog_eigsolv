from mpi4py import MPI
import pb11_time_scalapack_kernels
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

# =================
# output parameters
# =================
outfile = 'pdgemm_blocksize_vs_time.dat'

# ========================
# kernel launch parameters
# ========================
scale_factor = 3
min_blk_size = 2
mb_list = [min_blk_size * 2**x for x in range(scale_factor)]
nb_list = [min_blk_size * 2**x for x in range(scale_factor)]

niter = 1
nprow = 2
npcol = 2
m = nprow * mb_list[-1]
n = npcol * nb_list[-1]

# ============
# Input arrays
# ============
if (mpi_rank == 0):
    a = np.arange(m, dtype=np.float64)
    a = np.tile(a, (m, 1)).flatten()
    b = np.arange(m, dtype=np.float64).reshape(m, 1)
    b = np.tile(b, (1, m)).flatten()
    c = 0 * a
else:
    a = np.array([0], dtype=np.float64)
    b = np.array([0], dtype=np.float64)
    c = np.array([0], dtype=np.float64)

# [{'block': (mb, nb), time: t}]

t_kernel = []
for mb, nb in zip(mb_list, nb_list):

    t_tmp = pb11_time_scalapack_kernels.pb11_time_scalapack_pdgemm(
        niter, nprow, npcol, a, b, c, m, n, mb, nb)
    t_kernel.append(t_tmp)
t_kernel = mpi_comm.gather(t_kernel, root=0)


if mpi_rank == 0:
    print(t_kernel)

    df = pd.DataFrame(t_kernel, columns=[x for x in mb_list])
    print(df)
    df.to_csv(outfile, sep='\t', index=True, float_format='%.6f')

    # fig, ax = plt.subplots()
    # for i in range(mpi_size):
    #     ax.plot(mb_list, t_kernel[i], linestyle='--',
    #             marker='o', label="rank " + str(i))

    # ax.set_xlabel("block size")
    # ax.set_ylabel("time (s)")
    # ax.legend()
    # plt.show()
