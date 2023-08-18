#include <iostream>
// #include <cstring>
#include "mpi_hello.hpp"

void mpi_print_info(MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    std::cout << "size: " << size << " rank: " << rank << std::endl;
}

// void mpi_print_info()
// {
//     int argc{0};
//     char **argv = new char *[1];
//     argv[0] = new char[1];
//     strcpy(argv[0], "x");

//     MPI_Init(&argc, &argv);

//     int my_rank, mpi_size;
//     MPI_Comm_size(mpi_com, &mpi_size);
//     MPI_Comm_rank(mpi_com, &my_rank);

//     for (int irank = 0; irank < mpi_size; irank++)
//     {
//         if (irank == my_rank)
//         {
//             std::cout << "mpi_size: " << mpi_size << " ";
//             std::cout << "my_rank: " << my_rank << std::endl;
//         }
//         MPI_Barrier(mpi_com);
//     }

//     MPI_Finalize();
// }

// void mpi_my_rank(int &mpi_rank)
// {
//     int argc{0};
//     char **argv = new char *[1];
//     argv[0] = new char[1];
//     strcpy(argv[0], "x");

//     MPI_Init(&argc, &argv);

//     int my_rank, mpi_size;
//     MPI_Comm_size(mpi_com, &mpi_size);
//     MPI_Comm_rank(mpi_com, &my_rank);

//     mpi_rank = my_rank;

//     for (int irank = 0; irank < mpi_size; irank++)
//     {
//         if (irank == my_rank)
//         {
//             std::cout << "mpi_size: " << mpi_size << " ";
//             std::cout << "my_rank: " << my_rank << std::endl;
//         }
//         MPI_Barrier(mpi_com);
//     }

//     MPI_Finalize();
// }

// int main(int argc, char **argv)
// {

//     MPI_Init(&argc, &argv);

//     int my_rank, mpi_size;
//     MPI_Comm_size(mpi_com, &mpi_size);
//     MPI_Comm_rank(mpi_com, &my_rank);

//     for (int irank = 0; irank < mpi_size; irank++)
//     {
//         if (irank == my_rank)
//         {
//             std::cout << "mpi_size: " << mpi_size << " ";
//             std::cout << "my_rank: " << my_rank << std::endl;
//         }
//         MPI_Barrier(mpi_com);
//     }

//     MPI_Finalize();
//     return 0;
// }