#include <iostream>
#include "blacs.h"
#include <algorithm>

int main(int argc, char **argv)
{
    int nprow = atoi(argv[1]);
    int npcol = atoi(argv[2]);
    int m = atoi(argv[3]);
    int n = atoi(argv[4]);
    int mb = atoi(argv[5]);
    int nb = atoi(argv[6]);

    char barrier_all = 'A';
    char barrier_row = 'R';
    char barrier_col = 'C';

    // get context
    int ictxt;
    Cblacs_get(0, 0, &ictxt);

    // get process info
    int nprocs;
    int myrank;
    Cblacs_pinfo(&myrank, &nprocs);

    if (nprocs != nprow * npcol)
    {
        std::cerr << "Error: nprocs = " << nprocs << ", nprow * npcol = " << nprow * npcol << std::endl;
        Cblacs_gridexit(ictxt);
        Cblacs_exit(1);
    }

    // init process grid
    char order[] = "Row";
    Cblacs_gridinit(&ictxt, order, nprow, npcol);

    // get grid info
    int myrow;
    int mycol;
    Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);

    // get local data info
    int isrcproc = 0;
    int loc_m = numroc(&m, &mb, &myrow, &isrcproc, &nprow);
    int loc_n = numroc(&n, &nb, &mycol, &isrcproc, &npcol);

    // print info
    for (int i = 0; i < 100000000; i++)
    {
        for (int r = 0; r < myrank; r++)
        {
            double tmpval = 1.0 * 2.0;
        }
    }

    std::cout << "myrank = " << myrank << " ";
    std::cout << "myrow = " << myrow << " mycol = " << mycol << " ";
    std::cout << "loc_m = " << loc_m << " loc_n = " << loc_n;
    std::cout << std::endl;

    char scope = 'A';
    Cblacs_barrier(ictxt, &barrier_all);

    // init local descriptor
    int desc[9];
    int irsrc = 0;
    int icsrc = 0;
    int lld = std::max(1, loc_m);
    int info;
    descinit(desc, &m, &n, &mb, &nb, &irsrc, &icsrc, &ictxt, &lld, &info);

    // print desc info
    for (int r = 0; r < nprocs; r++)
    {
        if (myrank == r)
        {
            std::cout << "=========" << std::endl;
            std::cout << "myrank = " << myrank << " ";
            std::cout << "myrow = " << myrow << " mycol = " << mycol << " ";
            std::cout << std::endl;
            for (int i = 0; i < 9; i++)
            {
                std::cout << "desc[" << i << "] = " << desc[i] << std::endl;
            }
        }
        Cblacs_barrier(ictxt, &barrier_all);
    }

    // release context and exit
    Cblacs_gridexit(ictxt);
    Cblacs_exit(0);

    return 0;
}