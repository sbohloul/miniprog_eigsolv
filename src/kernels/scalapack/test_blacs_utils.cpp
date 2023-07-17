#include <iostream>
#include <cmath>
#include "blacs_utils.hpp"
#include "blacs.h"

int main()
{
    int myrank;
    int nprocs;
    Cblacs_pinfo(&myrank, &nprocs);

    int nprow = static_cast<int>(std::sqrt(nprocs));
    int npcol = nprocs / nprow;

    int ictxt_sys;
    Cblacs_get(-1, 0, &ictxt_sys);

    BlacsGrid grid(ictxt_sys, "Row", nprow, npcol);

    if (grid.get_context() < 0)
    {
        Cblacs_exit(1);
    }

    std::cout << "nprocs = " << grid.get_nprocs() << " "
              << "myrank = " << grid.get_myrank() << " "
              << "nprow = " << grid.get_nprow() << " "
              << "npcol = " << grid.get_npcol() << " "
              << "myrow = " << grid.get_myrow() << " "
              << "mycol = " << grid.get_mycol() << " "
              << std::endl;

    // Cblacs_gridexit(ictxt_sys);
    Cblacs_exit(0);

    return 0;
}