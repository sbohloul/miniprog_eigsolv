#include <iostream>
#include "blacs.h"
#include <algorithm>
#include <vector>

#define NPR 2
#define NPC 2
#define NROW 5
#define NCOL 5
#define NBROW 2
#define NBCOL 2

int main(int argc, char **argv)
{
    // int nprow = atoi(argv[1]);
    // int npcol = atoi(argv[2]);
    // int m = atoi(argv[3]);
    // int n = atoi(argv[4]);
    // int mb = atoi(argv[5]);
    // int nb = atoi(argv[6]);

    //
    int nprow = NPR;
    int npcol = NPC;
    int m = NROW;
    int n = NCOL;
    int mb = NBROW;
    int nb = NBCOL;

    // process grid order
    char grid_order[] = "Row";

    //
    char barrier_all = 'A';
    char barrier_row = 'R';
    char barrier_col = 'C';

    // get process info
    int nprocs;
    int myrank;
    Cblacs_pinfo(&myrank, &nprocs);

    // get default system context
    int ictxt_sys;
    Cblacs_get(0, 0, &ictxt_sys);

    // init proc grids
    int ictxt = ictxt_sys;
    Cblacs_gridinit(&ictxt, grid_order, nprow, npcol);

    if (ictxt < 0)
    {
        Cblacs_exit(1);
    }

    int ictxt_0 = ictxt_sys;
    Cblacs_gridinit(&ictxt_0, grid_order, 1, 1);

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
    Cblacs_barrier(ictxt, &barrier_all);

    // init local descriptor
    int desc[9];
    int irsrc = 0;
    int icsrc = 0;
    int lld = std::max(1, loc_m);
    int info;
    descinit(desc, &m, &n, &mb, &nb, &irsrc, &icsrc, &ictxt, &lld, &info);

    int desc_0[9];
    if (myrank == 0)
    {
        descinit(desc_0, &m, &n, &m, &n, &irsrc, &icsrc, &ictxt_0, &m, &info);
    }
    else
    {
        desc_0[1] = -1;
    }

    // print desc info

    // release context and exit
    Cblacs_gridexit(ictxt);
    Cblacs_exit(0);

    return 0;
}