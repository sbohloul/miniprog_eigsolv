#include <iostream>
#include <blacs.h>

void wait()
{
    for (int i = 0; i < 100000000; i++)
    {
    }
}

int main()
{
    int blacs_exit_zero = 0;
    int blacs_exit_one = 0;

    char order[] = "Row";

    // get context
    int ctxt_sys;
    Cblacs_get(0, 0, &ctxt_sys);

    int m = 5;
    int n = 5;
    int mb = 2;
    int nb = 2;
    int rsrcproc = 0;
    int rsrc = 0;
    int csrc = 0;

    //
    // grid 1
    //
    int nprow1 = 2;
    int npcol1 = 2;

    int myrank1, nprocs1;
    Cblacs_pinfo(&myrank1, &nprocs1);

    int ctxt_1 = ctxt_sys;
    Cblacs_gridinit(&ctxt_1, order, nprow1, npcol1);

    int myrow1, mycol1;
    Cblacs_gridinfo(ctxt_1, &nprow1, &npcol1, &myrow1, &mycol1);

    int ml = numroc_(&m, &mb, &myrow1, &rsrcproc, &nprow1);
    int nl = numroc_(&n, &nb, &mycol1, &rsrcproc, &npcol1);

    std::cout << "rank = " << myrank1 << " "
              << "myrow = " << myrow1 << " "
              << "mycol = " << mycol1 << " "
              << "ctxt_1 = " << ctxt_1 << " "
              << "ml = " << ml << " "
              << "nl = " << nl << " "
              << std::endl;

    if (ctxt_1 < 0)
    {
        std::cout << "rank = " << myrank1 << " not in the grid";
        Cblacs_exit(1);
    }

    wait();

    int info;
    int desc[9];
    int lld = std::max(1, ml);
    desc[1] = -1;
    if (~ctxt_1 < 0)
    {
        descinit_(desc, &m, &n, &mb, &nb, &rsrc, &csrc, &ctxt_1, &lld, &info);
    }
    // wait();

    //
    // grid 2
    //
    int nprow2 = 1;
    int npcol2 = 1;

    int myrank2, nprocs2;
    Cblacs_pinfo(&myrank2, &nprocs2);

    int ctxt_2 = ctxt_sys;
    Cblacs_gridinit(&ctxt_2, order, nprow2, npcol2);

    int myrow2, mycol2;
    Cblacs_gridinfo(ctxt_2, &nprow2, &npcol2, &myrow2, &mycol2);

    std::cout << "rank = " << myrank2 << " "
              << "myrow = " << myrow2 << " "
              << "mycol = " << mycol2 << " "
              << "ctxt_2 = " << ctxt_2 << " "
              << std::endl;

    // release context and finalize
    // Cblacs_gridexit(ctxt_sys);
    Cblacs_exit(0);

    return 0;
}