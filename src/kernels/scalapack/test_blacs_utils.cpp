#include <iostream>
#include <cmath>
#include "blacs_utils.hpp"
#include "blacs.h"

void test_blacs_grid(int ictxt)
{
    int myrank;
    int nprocs;
    Cblacs_pinfo(&myrank, &nprocs);

    int nprow = static_cast<int>(std::sqrt(nprocs));
    int npcol = nprocs / nprow;

    BlacsGrid grid(ictxt, "Row", nprow, npcol);

    if (grid.get_context() < 0)
    {
        Cblacs_exit(1);
        return;
    }

    std::cout << "nprocs = " << grid.get_nprocs() << " "
              << "myrank = " << grid.get_myrank() << " "
              << "nprow = " << grid.get_nprow() << " "
              << "npcol = " << grid.get_npcol() << " "
              << "myrow = " << grid.get_myrow() << " "
              << "mycol = " << grid.get_mycol() << " "
              << std::endl;

    // Cblacs_gridexit(ictxt);
}

void test_blacs_grid_ptr(int ictxt)
{
    int myrank;
    int nprocs;
    Cblacs_pinfo(&myrank, &nprocs);

    int nprow = static_cast<int>(std::sqrt(nprocs));
    int npcol = nprocs / nprow;

    BlacsGrid grid(ictxt, "Row", nprow, npcol);

    BlacsGrid *grid_ptr = &grid;

    if (grid_ptr->get_context() < 0)
    {
        Cblacs_exit(1);
        return;
    }

    std::cout << "nprocs = " << grid_ptr->get_nprocs() << " "
              << "myrank = " << grid_ptr->get_myrank() << " "
              << "nprow = " << grid_ptr->get_nprow() << " "
              << "npcol = " << grid_ptr->get_npcol() << " "
              << "myrow = " << grid_ptr->get_myrow() << " "
              << "mycol = " << grid_ptr->get_mycol() << " "
              << std::endl;

    // Cblacs_gridexit(ictxt);
}

void test_blacs_distarray(int &ictxt)
{
    int nprow = 2;
    int npcol = 3;
    BlacsGrid grid(ictxt, "Row", nprow, npcol);

    if (grid.get_context() < 0)
    {
        Cblacs_exit(1);
        return;
    }

    BlacsGrid *grid_ptr = &grid;

    int m = 5;
    int n = 6;
    int mb = 2;
    int nb = 2;

    BlacsDistributedArray dista = BlacsDistributedArray(grid_ptr, m, n, mb, nb);

    std::cout
        // << "dtype = " << dista.get_desctype() << " "
        << "ctxt = " << dista.get_context() << " "
        << "m = " << dista.global_size_row() << " "
        << "n = " << dista.global_size_col() << " "
        << "mb = " << dista.block_size_row() << " "
        << "nb = " << dista.block_size_col() << " "
        << "rsrc = " << dista.get_row_src() << " "
        << "csrc = " << dista.get_col_src() << " "
        << "lld = " << dista.get_lld() << " "
        << "ml = " << dista.local_size_row() << " "
        << "nl = " << dista.local_size_col() << " "
        << "info = " << dista.get_info() << " "

        << std::endl;

    int desca[9];
    dista.get_desc(desca);

    Cblacs_barrier(ictxt, "All");

    if (grid.get_myrank() == 0)
    {
        for (int i = 0; i < 9; i++)
        {
            std::cout << "desca[" << i << "]" << desca[i] << std::endl;
        }
    }
}

int main()
{
    int ictxt_sys;
    Cblacs_get(-1, 0, &ictxt_sys);

    std::cout << "Call test_blacs_grid" << std::endl;
    test_blacs_grid(ictxt_sys);
    Cblacs_barrier(ictxt_sys, "All");
    std::cout << std::endl;

    std::cout << "Call test_blacs_grid_ptr" << std::endl;
    test_blacs_grid_ptr(ictxt_sys);
    Cblacs_barrier(ictxt_sys, "All");
    std::cout << std::endl;

    test_blacs_distarray(ictxt_sys);
    Cblacs_barrier(ictxt_sys, "All");

    Cblacs_gridexit(ictxt_sys);
    Cblacs_exit(0);

    return 0;
}
