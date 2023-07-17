#include <iostream>
#include "blacs_utils.hpp"
#include "blacs.h"

BlacsGrid::BlacsGrid(int ictxt, char *order, int nprow, int npcol) : ctxt_(ictxt), order_(order), nprow_(nprow), npcol_(npcol)
{
    Cblacs_pinfo(&myrank_, &nprocs_);
    Cblacs_get(-1, 0, &ctxt_);
    Cblacs_gridinit(&ctxt_, order_, nprow_, npcol_);
    Cblacs_gridinfo(ctxt_, &nprow_, &npcol_, &myrow_, &mycol_);
}