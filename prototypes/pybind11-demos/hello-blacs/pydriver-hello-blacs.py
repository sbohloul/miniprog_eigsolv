from mpi4py import MPI
# import _pb11_hello_blacs

comm = MPI.COMM_WORLD
myrank = comm.Get_rank()
comm_size = comm.Get_size()

for r in range(comm_size):
    if r == myrank:
        print("myrank: ", myrank)
        comm.Barrier()

nprow = 2
npcol = 2
mb = 2
nb = 2
m = 5
n = 5


# _pb11_hello_blacs.print_info(nprow, npcol, mb, nb, m, n)
