#include <mpi.h>
#include <iostream>

int main(int argc, char **argv)
{
    MPI_Comm comm;
    MPI_Init(&argc, &argv);

    int size;
    MPI_Comm_size(comm, &size);

    int rank;
    MPI_Comm_rank(comm, &rank);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    if (rank == 0)
    {
        std::cout << "Number of processes: " << size << std::endl;
    }

    for (int i = 0; i < size; i++)
    {
        if (i == rank)
        {
            std::cout << "name = " << processor_name
                      << " rank = " << rank
                      << std::endl;
        }
        MPI_Barrier(comm);
    }

    MPI_Finalize();
    return 0;
}