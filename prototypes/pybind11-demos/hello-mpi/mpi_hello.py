# import mpi_hello

# # mpi_hello.mpi_print_info()

# my_rank = 0
# mpi_hello.mpi_my_rank(my_rank)
# print(my_rank)

from mpi4py import MPI

comm = MPI.COMM_WORLD
myrank = comm.Get_rank()
comm_size = comm.Get_size() 

for irank in range(comm_size):
    if (myrank == irank):
        print("size: ", comm_size, " myrank: ", myrank)
    comm.Barrier()
