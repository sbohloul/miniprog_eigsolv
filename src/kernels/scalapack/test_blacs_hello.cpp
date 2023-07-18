#include <iostream>
#include <cmath>
#include "blacs_utils.hpp"
#include "blacs.h"

int main()
{

    int myrank;
    int nprocs;
    Cblacs_pinfo(&myrank, &nprocs);

    int nprow = static_cast<int>(std::sqrt(nprocs));
    int npcol = nprocs / nprow;

    int ictxt_sys;
    Cblacs_get(-1, 0, &ictxt_sys);

    BlacsGrid grid(ictxt_sys, "Row", nprow, npcol);

    // if not in the grid, exit
    if (grid.get_context() < 0)
    {
        Cblacs_exit(1);
    }

    int ictxt = grid.get_context();
    int icaller = grid.get_myrank();
    int m = 1;
    int n = 1;
    int lda = 1;
    int sender_row;
    int sender_col;

    if (grid.get_myrank() == 0) // receive message
    {
        for (int i = 0; i < nprow; i++)
        {
            for (int j = 0; j < npcol; j++)
            {

                // std::cout << "myrank = " << grid.get_myrank() << " i = " << i << " j = " << j << std::endl;
                // receive msg
                if (i != 0 || j != 0)
                {
                    // std::cout << "igerv2d" << std::endl;

                    igerv2d(&ictxt, &m, &n, &icaller, &lda, &i, &j);
                }

                // std::cout << "myrank = " << grid.get_myrank() << "icaller = " << icaller << std::endl;

                // check if icaller is compatible with the grid
                blacs_pcoord(&ictxt, &icaller, &sender_row, &sender_col);

                // std::cout << "sender_row = " << sender_row << "sender_col = " << sender_col << std::endl;

                if (i != sender_row || j != sender_col)
                {
                    std::cerr << "Grid error, exiting ..." << std::endl;
                    return (1);
                }

                // print info
                std::cout << "i = " << i << " j = " << j << " icaller = " << icaller << std::endl;
            }
        }
    }
    else // send message
    {
        int rsrc = 0;
        int csrc = 0;
        // std::cout << "myrank = " << grid.get_myrank() << " icaller = " << icaller << std::endl;

        igesd2d(&ictxt, &m, &n, &icaller, &lda, &rsrc, &csrc);
    }

    // Cblacs_gridexit(ictxt_sys);
    Cblacs_exit(0);

    return 0;
}