#include <iostream>
#include "blacs.h"
#include "blacs_utils.hpp"

void print_array(double *a, int m, int n, int ld)
{

    for (int i = 0; i < m; i++)
    {
        std::cout << "[";
        for (int j = 0; j < n; j++)
        {
            std::cout << a[j * ld + i];
            if (j != n - 1)
            {
                std::cout << ", ";
            }
            else
            {
                std::cout << "]" << std::endl;
            }
        }
    }
}

double matval(int i, int j)
{
    return i + j / 10.0;
}

void wait(int rank)
{
    double tmpval = 1.0;
    for (int i = 0; i < 100000000 * (rank + 1); i++)
    {
        tmpval *= 1.0;
    }
}

int main()
{
    char barrier[] = "A";
    char order[] = "Row";
    int blacs_exit_zero = 0;
    int blacs_exit_one = 1;

    int ctxt_sys;
    Cblacs_get(-1, 0, &ctxt_sys);

    //
    // grid 1
    //
    int nprow = 2;
    int npcol = 2;
    BlacsGrid grid_2b2 = BlacsGrid(ctxt_sys, order, nprow, npcol);

    int rank = grid_2b2.get_myrank();

    std::cout << "rank = " << grid_2b2.get_myrank() << " "
              << "ctxt = " << grid_2b2.get_context() << " "
              << "nprow = " << grid_2b2.get_nprow() << " "
              << "npcol = " << grid_2b2.get_npcol() << " "
              << "myrow = " << grid_2b2.get_myrow() << " "
              << "mycol = " << grid_2b2.get_mycol() << " "
              << std::endl;
    wait(grid_2b2.get_myrank());

    // array
    int m = 5;
    int n = 5;
    int mb = 2;
    int nb = 2;
    BlacsArrayDescription bdesc_a = BlacsArrayDescription(&grid_2b2, m, n, mb, nb);

    std::cout << "rank = " << bdesc_a.grid()->get_myrank() << " "
              << "ctxt = " << bdesc_a.grid()->get_context() << " "
              << "myrow = " << bdesc_a.grid()->get_myrow() << " "
              << "mycol = " << bdesc_a.grid()->get_mycol() << " "
              << "ml = " << bdesc_a.local_size_row() << " "
              << "nl = " << bdesc_a.local_size_col() << " "
              << "lld = " << bdesc_a.get_lld() << " "
              << "info = " << bdesc_a.get_info() << " "
              << std::endl;

    wait(grid_2b2.get_myrank());

    //
    // grid 2
    //
    BlacsGrid grid_0 = BlacsGrid(ctxt_sys, order, 1, 1);

    std::cout << "rank = " << grid_0.get_myrank() << " "
              << "ctxt = " << grid_0.get_context() << " "
              << "nprow = " << grid_0.get_nprow() << " "
              << "npcol = " << grid_0.get_npcol() << " "
              << "myrow = " << grid_0.get_myrow() << " "
              << "mycol = " << grid_0.get_mycol() << " "
              << std::endl;
    wait(grid_0.get_myrank());

    BlacsArrayDescription bdesc_a_0 = BlacsArrayDescription(&grid_0, m, n, m, n);

    std::cout << "rank = " << bdesc_a_0.grid()->get_myrank() << " "
              << "cntxt = " << bdesc_a_0.grid()->get_context() << " "
              << "myrow = " << bdesc_a_0.grid()->get_myrow() << " "
              << "mycol = " << bdesc_a_0.grid()->get_mycol() << " "
              << "ml = " << bdesc_a_0.local_size_row() << " "
              << "nl = " << bdesc_a_0.local_size_col() << " "
              << "lld = " << bdesc_a_0.get_lld() << " "
              << "info = " << bdesc_a_0.get_info() << " "
              << std::endl;

    //
    DistributedArray arr_a = DistributedArray(&bdesc_a);

    for (int i = 0; i < arr_a.size(); i++)
    {
        arr_a.data()[i] = static_cast<double>(i);
    }

    wait(grid_0.get_myrank());

    std::cout << "rank = " << arr_a.get_grid()->get_myrank() << " "
              << "arr_a.size = " << arr_a.size() << std::endl;

    wait(grid_0.get_myrank());

    DistributedArray arr_a_0 = DistributedArray(&bdesc_a_0);
    int lld = arr_a_0.get_lld();
    for (int j = 0; j < arr_a_0.get_ncol(); j++)
    {
        for (int i = 0; i < arr_a_0.get_nrow(); i++)
        {
            arr_a_0.data()[i + j * lld] = matval(i, j);
        }
    }

    wait(grid_0.get_myrank());

    // redistribute a to b

    int desca_0[9];
    arr_a_0.get_desc()->get_desc(desca_0);

    int desca[9];
    arr_a.get_desc()->get_desc(desca);

    int ctxt = arr_a.get_grid()->get_context();

    if (~ctxt < 0)
    {
        Cpdgemr2d(m, n, arr_a_0.data(), 1, 1, desca_0,
                  arr_a.data(), 1, 1, desca,
                  arr_a.get_grid()->get_context());
    }
    std::cout << "myrank = " << rank << " arr_a: " << std::endl;

    print_array(arr_a.data(),
                arr_a.get_nrow(),
                arr_a.get_ncol(), arr_a.get_lld());

    // std::cout << "rank = " << arr_a.get_grid()->get_myrank() << " "
    //           << "arr_a_0.data = " << arr_a_0.data() << std::endl;

    // Cblacs_gridexit(ctxt_sys);
    Cblacs_exit(0);
    return 0;
}