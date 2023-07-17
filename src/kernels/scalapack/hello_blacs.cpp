#include <iostream>
#include <mpi.h>
#include <cassert>
#include <algorithm>
#include "blacs.h"

// extern "C"
// {
//     void Cblacs_get(int icontxt, int what, int *val);
//     void Cblacs_gridinit(int *icontxt, char *order, int nprow, int npcol);

//     void Cblacs_pinfo(int *rank, int *nprocs);
//     void Cblacs_gridinfo(int icontxt, int *nprow, int *npcol, int *myrow, int *mycol);

//     void Cblacs_gridexit(int icontxt);
//     void Cblacs_exit(int doneflag);
// }

int main(int argc, char **argv)
{

    int nprow = atoi(argv[1]);
    int npcol = atoi(argv[2]);
    int M = atoi(argv[3]);
    int N = atoi(argv[4]);
    int MB = atoi(argv[5]);
    int NB = atoi(argv[6]);

    // initialize the blacs environment and the grid
    int ictxt;
    Cblacs_get(0, 0, &ictxt);

    // get processes information
    int myrank;
    int nprocs;
    Cblacs_pinfo(&myrank, &nprocs);

    if (nprocs != nprow * npcol)
    {
        std::cerr << "Error: nprocs = " << nprocs << ", nprow * npcol = " << nprow * npcol << std::endl;
        Cblacs_gridexit(ictxt);
        Cblacs_exit(1);
    }

    // initialize the blacs environment and the grid
    // int ictxt;
    // Cblacs_get(0, 0, &ictxt);
    char order[] = "Row";
    Cblacs_gridinit(&ictxt, order, nprow, npcol);

    // get information about the grid and local processes
    int myrow, mycol;
    Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);

    for (int i = 0; i < 100 * myrank; i++)
    {
        int tmp = 1 * 1;
    }

    //
    // int M = 5;
    // int N = 5;
    // int MB = 2;
    // int NB = 2;
    int isrcproc = 0;
    int loc_m = numroc(&M, &MB, &myrow, &isrcproc, &nprow);
    int loc_n = numroc(&N, &NB, &mycol, &isrcproc, &npcol);

    std::cout << "myrow = " << myrow << " mycol = " << mycol << " myrank = " << myrank;
    std::cout << " nprow = " << nprow << " npcol = " << npcol << " nprocs = " << nprocs;
    std::cout << " loc_m = " << loc_m << " loc_n = " << loc_n;
    std::cout << std::endl;

    //
    int desc[9];
    int irsrc = 0;
    int icsrc = 0;
    int lld = std::max(1, loc_n); // column major
    int info;

    descinit(&desc, &M, &N, &MB, &NB, &irsrc, &icsrc, &ictxt, &lld, &info);

    // release the context and exit
    Cblacs_gridexit(ictxt);
    Cblacs_exit(0);

    return 0;
}