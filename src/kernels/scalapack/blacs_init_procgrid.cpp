#include <iostream>
#include <blacs.h>

int main(int argc, char **argv)
{
    constexpr int exit_normal = 0;
    constexpr int exit_error_grid = 1;

    int nprow = atoi(argv[1]);
    int npcol = atoi(argv[2]);

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
        Cblacs_exit(exit_error_grid);
    }

    // init grid
    char order[] = "Row";
    Cblacs_gridinit(&ictxt, order, nprow, npcol);

    // get grid info
    int myrow;
    int mycol;
    Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);

    // print info
    for (int i = 0; i < 100000000; i++)
    {
        for (int r = 0; r < myrank; r++)
        {
            double tmpval = 1.0 * 2.0;
        }
    }

    std::cout << "nprocs = " << nprocs << " myrank = " << myrank << " ";
    std::cout << "myrow = " << myrow << " mycol = " << mycol;
    std::cout << std::endl;

    // release context and finalize
    Cblacs_gridexit(ictxt);
    Cblacs_exit(exit_normal);

    return 0;
}