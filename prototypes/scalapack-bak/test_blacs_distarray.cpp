#include <iostream>
#include "blacs.h"
#include "blacs_utils.hpp"

int main()
{

    int ictxt_sys;
    Cblacs_get(-1, 0, &ictxt_sys);

    int nprow = 3;
    int npcol = 2;
    BlacsGrid pgrida = BlacsGrid(ictxt_sys, "Row", nprow, npcol);

    if (pgrida.get_context() < 0)
    {
        Cblacs_exit(1);
        return 1;
    }

    blacs_gridexit(&ictxt_sys);
    Cblacs_exit(0);
    return 0;
}