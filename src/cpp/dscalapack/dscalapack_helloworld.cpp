#include <iostream>
#include <cmath>

extern "C"
{
    void Cblacs_pinfo(int *, int *);
    void Cblacs_exit(int);
    void Cblacs_get(int, int, int *);
    void Cblacs_gridinit(int *, char const *, int, int);
    void Cblacs_gridinfo(int, int *, int *, int *, int *);
    int Cblacs_pnum(int, int, int);
    void Cigerv2d(int, int, int, int *, int, int, int);
    void Cigesd2d(int, int, int, int *, int, int, int);
    void Cblacs_pcoord(int, int, int *, int *);
}

int main(int argc, char **argv)
{

    // Number of processes and my process number
    int nprocs, myrank;
    Cblacs_pinfo(&myrank, &nprocs);

    // Define process grid, as close to square as possible
    int nprow = (int)sqrt(nprocs);
    int npcol = nprocs / nprow;

    if (myrank == 0)
    {
        std::cout << "Total number of processes: " << nprocs << std::endl;
        std::cout << "process grid: {" << nprow << "," << npcol << "}" << std::endl;
    }

    // Get default system context and initialize the process grid
    int ictxt;
    Cblacs_get(0, 0, &ictxt);
    Cblacs_gridinit(&ictxt, "Row-major", nprow, npcol);

    // Get grid info
    int myprow, mypcol;
    Cblacs_gridinfo(ictxt, &nprow, &npcol, &myprow, &mypcol);

    // Only processes in the grid enters this block
    if (myprow >= 0 && mypcol >= 0)
    {
        // Get my process number
        int icaller;
        icaller = Cblacs_pnum(ictxt, myprow, mypcol);

        // Process {0,0} receives check-in messages
        if (myprow == 0 && mypcol == 0)
        {
            int hisprow, hispcol;
            for (int irow = 0; irow < nprow; irow++)
            {
                for (int jcol = 0; jcol < npcol; jcol++)
                {
                    // receive info sent by {irow, jcol}
                    if (irow != 0 || jcol != 0)
                    {
                        Cigerv2d(ictxt, 1, 1, &icaller, 1, irow, jcol);
                    }

                    // Verify received info
                    Cblacs_pcoord(ictxt, icaller, &hisprow, &hispcol);
                    if (hisprow != irow && hispcol != jcol)
                    {
                        std::cout << "Error in process grid." << std::endl;
                        return 1;
                    }

                    std::cout
                        << "icaller: " << icaller
                        << " row: " << irow << " col: " << jcol << std::endl;
                }
            }
            std::cout << "All processes checked in." << std::endl;
        }
        else
        {
            Cigesd2d(ictxt, 1, 1, &icaller, 1, 0, 0);
        }
    }

    Cblacs_exit(0);

    return 0;
}