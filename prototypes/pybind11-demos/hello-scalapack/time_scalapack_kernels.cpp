#include <iostream>
#include <cassert>
#include "time_scalapack_kernels.hpp"
#include "blacs_.h"
#include "timer.hpp"

double time_scalapack_pdgemm(int niter, int nprow, int npcol, const std::vector<double> &a_glb, const std::vector<double> &b_glb, std::vector<double> &c_glb, int m, int n, int mb, int nb)
{
    assert(a_glb.size() == m * n);
    assert(b_glb.size() == m * n);
    assert(c_glb.size() == m * n);

    // general initializations
    char grid_order[] = "Row";
    int isrcproc = 0;
    int rsrc = 0;
    int csrc = 0;

    // processes info
    int myrank, nprocs;
    Cblacs_pinfo(&myrank, &nprocs);

    // init blacs contexts and grids
    int ctxt_sys;
    Cblacs_get(-1, 0, &ctxt_sys);
    int ctxt_all = ctxt_sys;
    Cblacs_gridinit(&ctxt_all, grid_order, 1, nprocs);
    int ctxt = ctxt_sys;
    Cblacs_gridinit(&ctxt, grid_order, nprow, npcol);
    int ctxt_0 = ctxt_sys;
    Cblacs_gridinit(&ctxt_0, grid_order, 1, 1);

    // grid info
    int myrow, mycol;
    Cblacs_gridinfo(ctxt, &nprow, &npcol, &myrow, &mycol);
    if (ctxt < 0)
    {
        std::cout << "***NOT IN GRID, process: " << myrank << std::endl;
    }

    // local array information
    int ml = numroc(&m, &mb, &myrow, &isrcproc, &nprow);
    int nl = numroc(&n, &nb, &mycol, &isrcproc, &npcol);
    int lld = std::max(1, ml);
    int info;
    int desc_loc[9];
    descinit(desc_loc, &m, &n, &mb, &nb, &rsrc, &csrc, &ctxt, &lld, &info);
    if (info < 0)
    {
        std::cerr << "***ERROR IN desc_loc, info: " << info << std::endl;
    }

    // global array information
    int desc_glb[9];
    desc_glb[1] = -1;
    if (myrank == 0)
    {
        descinit(desc_glb, &m, &n, &m, &n, &rsrc, &csrc, &ctxt_0, &m, &info);
        if (info < 0)
        {
            std::cerr << "***ERROR IN desc_glb, info: " << info << std::endl;
        }
    }

    // allocate local arrays
    std::vector<double> a_loc(ml * nl);
    std::vector<double> b_loc(ml * nl);
    std::vector<double> c_loc(ml * nl);

    // distribute global to local
    Cpdgemr2d(m, n,
              a_glb.data(), 1, 1, desc_glb,
              a_loc.data(), 1, 1, desc_loc,
              ctxt);
    Cpdgemr2d(m, n,
              b_glb.data(), 1, 1, desc_glb,
              b_loc.data(), 1, 1, desc_loc,
              ctxt);

    // call pdgemm kernel
    Timer timer;
    double alpha = 1.0;
    double beta = 0.0;
    int ia = 1, ja = 1;
    int ib = 1, jb = 1;
    int ic = 1, jc = 1;

    timer.start();
    for (int i = 0; i < niter; i++)
    {
        pdgemm("N", "N",
               &m, &n, &n,
               &alpha,
               a_loc.data(), &ia, &ja, desc_loc,
               b_loc.data(), &ib, &jb, desc_loc,
               &beta,
               c_loc.data(), &ic, &jc, desc_loc);
    }
    timer.stop();
    double t_kernel = timer.duration();
    // double t_kernel = 1.0;

    // gather local C in global C
    Cpdgemr2d(m, n,
              c_loc.data(), 1, 1, desc_loc,
              c_glb.data(), 1, 1, desc_glb,
              ctxt);

    // release blacs contexts
    Cblacs_gridexit(ctxt);
    // Cblacs_exit(1);

    return t_kernel;
}