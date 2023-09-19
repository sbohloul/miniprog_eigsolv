#include <iostream>
#include <cassert>
#include "mpistats_scalapack_kernels.hpp"

// ======
// PDSYEV
// ======
void mpistats_scalapack_pdsyev(int mpi_region, int nprow, int npcol, const std::vector<double> &a_glb, std::vector<double> &eigval_glb, std::vector<double> &eigvec_glb, int m, int mb, int nb)
{
    // Turn off mpi profiling
    MPI_Pcontrol(0);

    // processes info
    int myrank, nprocs;
    Cblacs_pinfo(&myrank, &nprocs);

    // verify input array dimensions
    if (myrank == 0)
    {
        assert(a_glb.size() == m * m);
        assert(eigvec_glb.size() == m * m);
        assert(eigval_glb.size() == m);
    }

    // general initializations
    char grid_order[] = "Row";
    int isrcproc = 0;
    int rsrc = 0;
    int csrc = 0;
    int info;

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
    int nl = numroc(&m, &nb, &mycol, &isrcproc, &npcol);
    int lld = std::max(1, ml);
    int desc_loc[9];
    descinit(desc_loc, &m, &m, &mb, &nb, &rsrc, &csrc, &ctxt, &lld, &info);
    if (info < 0)
    {
        std::cerr << "***ERROR IN desc_loc, info: " << info << std::endl;
    }

    // global array information
    int desc_glb[9];
    desc_glb[1] = -1;
    if (myrank == 0)
    {
        descinit(desc_glb, &m, &m, &m, &m, &rsrc, &csrc, &ctxt_0, &m, &info);
        if (info < 0)
        {
            std::cerr << "***ERROR IN desc_glb, info: " << info << std::endl;
        }
    }

    // allocate local arrays
    std::vector<double> a_loc(ml * nl);
    std::vector<double> eigvec_loc(ml * nl);

    // distribute global to local
    Cpdgemr2d(m, m,
              a_glb.data(), 1, 1, desc_glb,
              a_loc.data(), 1, 1, desc_loc,
              ctxt);

    // call pdsyev kernel
    const char jobz = 'V';
    const char uplo = 'U';
    int ia = 1, ja = 1;
    int ieigvec = 1, jeigvec = 1;
    int lwork;

    // optimal lwork
    lwork = -1;
    double work_optimized;
    pdsyev(&jobz, &uplo,
           &m, a_loc.data(), &ia, &ja, desc_loc,
           eigval_glb.data(),
           eigvec_loc.data(), &ieigvec, &jeigvec, desc_loc,
           &work_optimized, &lwork, &info);

    if (info < 0)
    {
        if (myrank == 0)
        {
            std::cerr << "***ERROR IN pdsyev lwork query, info: " << info << std::endl;
        }
    }

    // update lwork and work
    lwork = static_cast<int>(work_optimized);
    std::vector<double> work(lwork);

    // first run to check convergence problems
    pdsyev(&jobz, &uplo,
           &m, a_loc.data(), &ia, &ja, desc_loc,
           eigval_glb.data(),
           eigvec_loc.data(), &ieigvec, &jeigvec, desc_loc,
           work.data(), &lwork, &info);

    if (info > 0)
    {
        if (myrank == 0)
        {
            std::cerr << "***ERROR IN pdsyev first run, info: " << info << std::endl;
        }
    }

    // =================
    // collect mpi stats
    // =================
    if (myrank == 0)
    {
        std::cout << "mpi_region: " << mpi_region << std::endl;
    }

    // Reset a_loc
    Cpdgemr2d(m, m,
              a_glb.data(), 1, 1, desc_glb,
              a_loc.data(), 1, 1, desc_loc,
              ctxt);

    // Turn on mpi profiling
    MPI_Pcontrol(1);

    // Mark region and profile mpi
    int mpi_control = mpi_region;

    MPI_Pcontrol(mpi_control);

    pdsyev(&jobz, &uplo,
           &m, a_loc.data(), &ia, &ja, desc_loc,
           eigval_glb.data(),
           eigvec_loc.data(), &ieigvec, &jeigvec, desc_loc,
           work.data(), &lwork, &info);

    mpi_control = -mpi_region;
    MPI_Pcontrol(mpi_control);

    // Turn off profiling
    MPI_Pcontrol(0);

    // =================
    // =================

    // gather local eigvec in global eigvec
    Cpdgemr2d(m, m,
              eigvec_loc.data(), 1, 1, desc_loc,
              eigvec_glb.data(), 1, 1, desc_glb,
              ctxt);

    // release blacs contexts
    Cblacs_gridexit(ctxt);
}
