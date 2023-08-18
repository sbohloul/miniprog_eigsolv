#ifndef _MPI_HELLO_HPP_
#define _MPI_HELLO_HPP_

#include <mpi.h>

// void mpi_print_info();
// void mpi_my_rank(int &mpi_rank);

void mpi_print_info(MPI_Comm comm);

#endif