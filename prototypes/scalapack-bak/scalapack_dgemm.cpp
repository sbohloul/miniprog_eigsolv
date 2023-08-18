#include <iostream>
#include <vector>
#include "blacs.h"

#define M 5
#define N 5
#define MB 2
#define NB 2
#define NPROW 2
#define NPCOL 2

// **************
// util functions
// **************
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

int main()
{
    // distributed arrays information
    int m = M;
    int n = N;
    int mb = MB;
    int nb = NB;
    int isrcproc = 0;
    int rsrc = 0;
    int csrc = 0;

    // process info
    int myrank;
    int nprocs;
    Cblacs_pinfo(&myrank, &nprocs);

    // init blacs contexts
    int ctxt_sys;
    Cblacs_get(-1, 0, &ctxt_sys);

    // init process grid
    char pg_order[] = "Row";
    int nprow = NPROW;
    int npcol = NPCOL;
    int ctxt = ctxt_sys;
    Cblacs_gridinit(&ctxt, pg_order, nprow, npcol);
    int ctxt_0 = ctxt_sys;
    Cblacs_gridinit(&ctxt_0, pg_order, 1, 1);
    int ctxt_all = ctxt_sys;
    Cblacs_gridinit(&ctxt_all, pg_order, 1, nprocs);

    // get grid info
    int myrow;
    int mycol;
    Cblacs_gridinfo(ctxt, &nprow, &npcol, &myrow, &mycol);

    if (ctxt < 0)
    {
        std::cout << "process " << myrank << " not in the grid." << std::endl;
        // Cblacs_gridexit(ctxt);
        return 0;
    }

    // local arrays descriptor
    int ml = numroc(&m, &mb, &myrow, &isrcproc, &nprow);
    int nl = numroc(&n, &nb, &mycol, &isrcproc, &npcol);
    int desc_loc[9];
    int lld = std::max(1, ml);
    int info;
    descinit(desc_loc, &m, &n, &mb, &nb, &rsrc, &csrc, &ctxt, &lld, &info);

    if (info < 0)
    {
        std::cerr << "Error in descinit for desc_loc, info: " << info << std::endl;
    }

    // global array descriptor
    int desc_glb[9];
    desc_glb[1] = -1;
    if (myrank == 0)
    {
        descinit(desc_glb, &m, &n, &m, &n, &rsrc, &csrc, &ctxt_0, &m, &info);
        if (info < 0)
        {
            std::cerr << "Error in descinit for desc_glb, info: " << info << std::endl;
        }
    }

    // allocate local arrays
    std::vector<double> a_loc(ml * nl);
    std::vector<double> b_loc(ml * nl);
    std::vector<double> c_loc(ml * nl);

    // allocate global array on master process
    std::vector<double> a_glb;
    std::vector<double> b_glb;
    std::vector<double> c_glb;

    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < m; i++)
        {
            // arr_a_0[j * m + i] = matval(i, j);
            a_glb.push_back(static_cast<double>(j));
            b_glb.push_back(static_cast<double>(i));
            c_glb.push_back(static_cast<double>(0));
        }
    }

    if (myrank == 0)
    {
        std::cout << "a_glb: " << std::endl;
        print_array(a_glb, m, n, m);
        std::cout << "b_glb: " << std::endl;
        print_array(b_glb, m, n, m);
        std::cout << "c_glb: " << std::endl;
        print_array(c_glb, m, n, m);
    }
    blacs_barrier(&ctxt_all, "A");

    // distribute arrays
    Cpdgemr2d(m, n,
              a_glb.data(), 1, 1, desc_glb,
              a_loc.data(), 1, 1, desc_loc,
              ctxt);
    Cpdgemr2d(m, n,
              b_glb.data(), 1, 1, desc_glb,
              b_loc.data(), 1, 1, desc_loc,
              ctxt);

    // print local arrays
    std::cout << "myrank = " << myrank << " a_loc: " << std::endl;
    print_array(a_loc, ml, nl, lld);
    blacs_barrier(&ctxt_all, "A");
    std::cout << "myrank = " << myrank << " b_loc: " << std::endl;
    print_array(b_loc, ml, nl, lld);
    blacs_barrier(&ctxt_all, "A");

    // call pdgemm
    double alpha = 1.0;
    double beta = 0.0;
    int ia = 1;
    int ja = 1;
    int ib = 1;
    int jb = 1;
    int ic = 1;
    int jc = 1;
    pdgemm("N", "N",
           &m, &n, &n,
           &alpha,
           b_loc.data(), &ia, &ja, desc_loc,
           a_loc.data(), &ib, &jb, desc_loc,
           &beta,
           c_loc.data(), &ic, &jc, desc_loc);

    // gather distributed C in master
    Cpdgemr2d(m, n,
              c_loc.data(), 1, 1, desc_loc,
              c_glb.data(), 1, 1, desc_glb,
              ctxt);
    blacs_barrier(&ctxt_all, "A");

    std::cout << "myrank = " << myrank << " c_loc: " << std::endl;
    print_array(c_loc, ml, nl, lld);
    blacs_barrier(&ctxt_all, "A");

    // print global C
    if (myrank == 0)
    {
        std::cout << "c_glb: " << std::endl;
        print_array(c_glb, m, n, m);
    }
    blacs_barrier(&ctxt_all, "A");

    // finalize blacs
    Cblacs_exit(0);

    return 0;
}