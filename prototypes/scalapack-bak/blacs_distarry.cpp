#include <iostream>
#include <vector>
#include "blacs.h"

int blacs_continue_zero = 0;
char grid_order[] = "Row";

// ==============
// util functions
// ==============
double matval(int i, int j)
{
    return i + j / 10.0;
}

void print_array(std::vector<double> a, int m, int n, int ld)
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

void wait(int factor)
{
    int tmp_val = 1;
    for (int i = 0; i < 10000000 * factor; i++)
    {
        tmp_val += 1 + 1 + 1;
    }
}

int indxl2g(int indxloc, int nb, int iproc, int isrcproc, int nprocs)
{
    indxloc += 1;
    return indxl2g_(&indxloc, &nb, &iproc, &isrcproc, &nprocs) - 1;
}

// =======
// drivers
// =======
void set_distributed_array(int ctxt, int myrank, int nprocs)
{
    // initialize grid
    int nprow = 2;
    int npcol = 2;
    Cblacs_gridinit(&ctxt, grid_order, nprow, npcol);
    if (ctxt < 0)
    {
        std::cout << "proc " << myrank << " not in the grid" << std::endl;
        // Cblacs_exit(0);
        // blacs_exit(&blacs_continue_zero);
        // Cblacs_gridexit(ctxt);
        return;
    }

    // grid info
    int myrow;
    int mycol;
    Cblacs_gridinfo(ctxt, &nprow, &npcol, &myrow, &mycol);

    // array  a info
    int isrcproc = 0;
    int m = 5;
    int n = 5;
    int mb = 2;
    int nb = 2;
    int ml = numroc(&m, &mb, &myrow, &isrcproc, &nprow);
    int nl = numroc(&n, &nb, &mycol, &isrcproc, &npcol);

    // print grid info
    std::cout
        << "myrank = " << myrank << " "
        << "nprocs = " << nprocs << " "
        << "myrow = " << myrow << " "
        << "mycol = " << mycol << " "
        << "ml = " << ml << " "
        << "nl = " << nl << " "
        << std::endl;

    // array descriptors
    int desca[9];
    int rsrc = 0;
    int csrc = 0;
    int lld = std::max(1, ml);
    int info;
    descinit(desca, &m, &n, &mb, &nb, &rsrc, &csrc, &ctxt, &lld, &info);

    // allocate array & set elements
    std::vector<double> arr_a(ml * nl);

    int i, j;
    for (int jl = 0; jl < nl; jl++)
    {
        j = indxl2g(jl, nb, mycol, isrcproc, npcol);
        for (int il = 0; il < ml; il++)
        {
            i = indxl2g(il, mb, myrow, isrcproc, nprow);

            std::cout << "myrank = " << myrank << " "
                      << "il, jl = " << il << ", " << jl << " "
                      << "i, j = " << i << ", " << j
                      << std::endl;

            arr_a[jl * lld + il] = matval(i, j);
        }
    }
    blacs_barrier(&ctxt, "All");

    // print local array
    wait(myrank);
    std::cout << "myrank = " << myrank << " myarray: " << std::endl;
    print_array(arr_a, ml, nl, lld);
    blacs_barrier(&ctxt, "All");
}

void redist_change_blocksize(int ctxt, int myrank, int nprocs)
{

    int nprow = 2;
    int npcol = 2;
    Cblacs_gridinit(&ctxt, grid_order, nprow, npcol);
    if (ctxt < 0)
    {
        std::cout << "proc " << myrank << " not in the grid" << std::endl;
        return;
    }

    // grid info
    int myrow;
    int mycol;
    Cblacs_gridinfo(ctxt, &nprow, &npcol, &myrow, &mycol);

    // array  a info
    int isrcproc = 0;
    int m = 5;
    int n = 5;
    int mb_a = 2;
    int nb_a = 2;
    int ml_a = numroc(&m, &mb_a, &myrow, &isrcproc, &nprow);
    int nl_a = numroc(&n, &nb_a, &mycol, &isrcproc, &npcol);

    // array a descriptors
    int desca[9];
    int rsrc = 0;
    int csrc = 0;
    int lld_a = std::max(1, ml_a);
    int info;
    descinit(desca, &m, &n, &mb_a, &nb_a, &rsrc, &csrc, &ctxt, &lld_a, &info);

    // allocate array & set elements
    std::vector<double> arr_a(ml_a * nl_a);
    int i, j;
    for (int jl = 0; jl < nl_a; jl++)
    {
        j = indxl2g(jl, nb_a, mycol, isrcproc, npcol);
        for (int il = 0; il < ml_a; il++)
        {
            i = indxl2g(il, mb_a, myrow, isrcproc, nprow);
            arr_a[jl * lld_a + il] = matval(i, j);
        }
    }

    // array b info and descriptor
    int mb_b = 1;
    int nb_b = 2;
    int ml_b = numroc(&m, &mb_b, &myrow, &isrcproc, &nprow);
    int nl_b = numroc(&n, &nb_b, &mycol, &isrcproc, &npcol);

    int descb[9];
    int lld_b = std::max(1, ml_b);
    descinit(descb, &m, &n, &mb_b, &nb_b, &rsrc, &csrc, &ctxt, &lld_b, &info);

    // allocate array b
    std::vector<double> arr_b(ml_b * nl_b);

    // redistribute a to b
    Cpdgemr2d(m, n, arr_a.data(), 1, 1, desca,
              arr_b.data(), 1, 1, descb,
              ctxt);

    wait(myrank);
    std::cout << "myrank = " << myrank << " arr_a: " << std::endl;
    print_array(arr_a, ml_a, nl_a, lld_a);
    std::cout << "myrank = " << myrank << " arr_b: " << std::endl;
    print_array(arr_b, ml_b, nl_b, lld_b);

    blacs_barrier(&ctxt, "All");
}

void dist_master_to_all(int ctxt, int myrank, int nprocs)
{

    int nprow = 2;
    int npcol = 2;
    Cblacs_gridinit(&ctxt, grid_order, nprow, npcol);
    if (ctxt < 0)
    {
        std::cout << "proc " << myrank << " not in the grid" << std::endl;
        return;
    }

    // grid info
    int myrow;
    int mycol;
    Cblacs_gridinfo(ctxt, &nprow, &npcol, &myrow, &mycol);

    // array a info
    int isrcproc = 0;
    int m = 5;
    int n = 5;
    int mb_a = 3;
    int nb_a = 3;
    int ml_a = numroc(&m, &mb_a, &myrow, &isrcproc, &nprow);
    int nl_a = numroc(&n, &nb_a, &mycol, &isrcproc, &npcol);

    // array a descriptors
    int desca[9];
    int rsrc = 0;
    int csrc = 0;
    int lld_a = std::max(1, ml_a);
    int info;
    descinit(desca, &m, &n, &mb_a, &nb_a, &rsrc, &csrc, &ctxt, &lld_a, &info);

    // allocate local array
    std::vector<double> arr_a(ml_a * nl_a);

    // master process and global array a
    int ctxt_0 = ctxt;
    Cblacs_gridinit(&ctxt_0, grid_order, 1, 1);

    int desca_0[9];
    desca_0[1] = -1;
    if (myrank == 0)
    {
        descinit(desca_0, &m, &n, &m, &n, &rsrc, &csrc, &ctxt_0, &m, &info);
    }

    // allocate global array on master process
    std::vector<double> arr_a_0;
    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < m; i++)
        {
            // arr_a_0[j * m + i] = matval(i, j);
            arr_a_0.push_back(matval(i, j));
        }
    }

    // redistribute a to b
    Cpdgemr2d(m, n, arr_a_0.data(), 1, 1, desca_0,
              arr_a.data(), 1, 1, desca,
              ctxt);

    wait(myrank);
    std::cout << "myrank = " << myrank << " arr_a: " << std::endl;
    print_array(arr_a, ml_a, nl_a, lld_a);
    blacs_barrier(&ctxt, "All");
}

int main()
{

    // process info
    int myrank;
    int nprocs;
    Cblacs_pinfo(&myrank, &nprocs);

    // initialize context
    int ctxt_sys;
    Cblacs_get(-1, 0, &ctxt_sys);

    //
    // set_distributed_array(ctxt_sys, myrank, nprocs);

    //
    // redist_change_blocksize(ctxt_sys, myrank, nprocs);

    //
    dist_master_to_all(ctxt_sys, myrank, nprocs);

    // exit grid and context
    // Cblacs_gridexit(ctxt_sys);
    Cblacs_exit(0);

    return 0;
}